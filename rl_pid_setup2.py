# RL-PID (Guan & Yamamoto) — Fixed PyTorch Version
# Explicit (non-autograd) update rules: Eq. (19)-(21), (23)-(25)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ------------------------------------------------------------
# Nonlinear plant
# ------------------------------------------------------------
def plant_step(y_prev, u, y_pprev, noise_std=0.01):
    noise = torch.randn(1, device=device) * noise_std
    y = (y_prev * y_pprev * (y_prev + 2.5)) / (1 + y_prev**2 + y_pprev**2) + u + noise
    return y.squeeze()

# ------------------------------------------------------------
# Desired reference
# ------------------------------------------------------------
def desired_response(t):
    if t < 100:
        return 2.5
    elif t < 200:
        return 5.0
    elif t < 300:
        return 4.0
    else:
        return 3.0

# ------------------------------------------------------------
# RBF activation (handles all shape mismatches safely)
# ------------------------------------------------------------
def rbf_activations(x, centers, sigmas):
    # x: (D,), centers: (M, D), sigmas: (M,)
    diff = centers - x.unsqueeze(0).expand_as(centers)  # (M, D)
    sq_dist = torch.sum(diff**2, dim=1)  # (M,)
    s = torch.clamp(sigmas, min=1e-8)
    phi = torch.exp(-sq_dist / (2.0 * s**2))
    return phi  # (M,)

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
T = 400
gamma = 0.98
noise_std = 0.01

M_actor = 8
M_critic = 8
state_dim_actor = 3
state_dim_critic = 3

alpha_v = 1e-4
alpha_mu = 1e-4
alpha_sigma = 1e-4
alpha_w = 1e-4
jac_sign = 1.0  # change to -1 if reverse-acting

# ------------------------------------------------------------
# Initialize RBF parameters
# ------------------------------------------------------------
def make_centers(M, dim, spread=1.0):
    return (2 * torch.rand((M, dim), device=device) - 1) * spread

centers_actor = make_centers(M_actor, state_dim_actor)
sigmas_actor = torch.ones(M_actor, device=device) * 0.6

centers_critic = make_centers(M_critic, state_dim_critic)
sigmas_critic = torch.ones(M_critic, device=device) * 0.6

# Initialize weights
wP = torch.randn(M_actor, device=device) * 1e-3
wI = torch.randn(M_actor, device=device) * 1e-3
wD = torch.randn(M_actor, device=device) * 1e-3
v = torch.randn(M_critic, device=device) * 1e-3

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def actor_gains(phi):
    KP = torch.dot(wP, phi)
    KI = torch.dot(wI, phi)
    KD = torch.dot(wD, phi)
    return KP, KI, KD

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------
y_hist, yd_hist = [], []
KP_hist, KI_hist, KD_hist = [], [], []
u_hist, td_hist, V_hist = [], [], []

# ------------------------------------------------------------
# Initial states
# ------------------------------------------------------------
y_pprev = torch.tensor(0.0, device=device)
y_prev = torch.tensor(0.0, device=device)
y = torch.tensor(0.0, device=device)
u_prev = torch.tensor(0.0, device=device)

# Initial critic value
e = torch.tensor(desired_response(0) - y.item(), device=device)
dy = torch.tensor(0.0, device=device)
ddy = torch.tensor(0.0, device=device)
psi_c = torch.stack([e, -dy, -ddy])
phi_c = rbf_activations(psi_c, centers_critic, sigmas_critic)
V_current = torch.dot(v, phi_c)

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
num_epochs = 200
for t in tqdm(range(T)):
    yd = torch.tensor(desired_response(t), device=device)
    e = yd - y
    dy = y - y_prev
    ddy = y - 2 * y_prev + y_pprev # second difference
    
    for epoch in range(num_epochs):
      # Actor RBFs
      psi_a = torch.stack([e, -dy, -ddy])
      phi_a = rbf_activations(psi_a, centers_actor, sigmas_actor)
      KP, KI, KD = actor_gains(phi_a)
      KP = torch.clamp(KP, -50.0, 50.0)
      KI = torch.clamp(KI, -10.0, 10.0)
      KD = torch.clamp(KD, -10.0, 10.0)

      # Incremental control (velocity form)
      u_inc = KI * e - KP * (y - y_prev) - KD * (y_prev - 2 * y_prev + y_pprev)
      u = u_prev + u_inc

      # Plant update
      y_next = plant_step(y, u, y_prev, noise_std)
      r = 0.5 * (yd - y_next) ** 2

      # Critic RBFs
      e_next = yd - y_next
      dy_next = y_next - y
      ddy_next = y_next - 2 * y + y_prev
      psi_c_next = torch.stack([e_next, -dy_next, -ddy_next])
      phi_c_next = rbf_activations(psi_c_next, centers_critic, sigmas_critic)
      V_next = torch.dot(v, phi_c_next)

      # TD error
      delta_TD = r + gamma * V_next - V_current

      # ------------- Critic update (Eqs. 23–25) -------------
      psi_c_cur = torch.stack([e, -dy, -ddy])
      phi_c_cur = rbf_activations(psi_c_cur, centers_critic, sigmas_critic)
      v = v + alpha_v * delta_TD * phi_c_cur

      for j in range(M_critic):
          phi_j = phi_c_cur[j]
          vj = v[j]
          diff = psi_c_cur - centers_critic[j]
          dist2 = torch.sum(diff**2)
          sig = torch.clamp(sigmas_critic[j], min=1e-6)
          centers_critic[j] = centers_critic[j] + alpha_mu * delta_TD * vj * phi_j * diff / (sig**2)
          sigmas_critic[j] = sigmas_critic[j] + alpha_sigma * delta_TD * vj * phi_j * dist2 / (sig**3)
          sigmas_critic[j] = torch.clamp(sigmas_critic[j], min=1e-4)

      # ------------- Actor update (Eqs. 19–21) -------------
      factor = abs((yd - y_next))*np.sign((yd - y_next))   # this multiplies the rest
      delta2_y = y - 2 * y_prev + y_pprev
      grad_wP = -delta_TD * factor * (y - y_prev) * phi_a
      grad_wI = -delta_TD * factor * e * phi_a
      grad_wD = -delta_TD * factor * delta2_y * phi_a
      wP = wP - alpha_w * grad_wP
      wI = wI - alpha_w * grad_wI
      wD = wD - alpha_w * grad_wD

    # Shift variables
    y_pprev, y_prev, y = y_prev.clone(), y.clone(), y_next.clone()
    u_prev = u.clone()
    V_current = V_next.clone()

    # Logs
    y_hist.append(y_next.item())
    yd_hist.append(yd.item())
    u_hist.append(u.item())
    KP_hist.append(KP.item())
    KI_hist.append(KI.item())
    KD_hist.append(KD.item())
    td_hist.append(delta_TD.item())
    V_hist.append(V_current.item())

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(y_hist, 'k', label='Output y')
plt.plot(yd_hist, 'r--', label='Desired yd')
plt.title("RL-PID (Fixed PyTorch Implementation)")
plt.xlabel("Time step")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# Plot results
# ---------------------------
fig = plt.figure(figsize=(12,6))
plt.suptitle('RL-PID (paper-style) Input and Output')

plt.subplot(2,1,1)
plt.plot(t_arr, y_hist, label='Output (y)', color='k')
plt.plot(t_arr, yd_hist, '--', label='Desired (yd)', color='r')
plt.xlabel('Time step')
plt.ylabel('$y$')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.subplot(2,1,2)
plt.plot(t_arr, u_hist, color='k')
plt.xlabel('Time step')
plt.ylabel('$u$')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.tight_layout()

# ---------------------------
# Plot results
# ---------------------------
fig = plt.figure(figsize=(12,6))
plt.suptitle('RL-PID (paper-style) TD Error ($\\delta_{TD}$) and V')

plt.subplot(2,1,1)
plt.plot(t_arr, td_hist, color='b')
plt.xlabel('Time step')
plt.ylabel('$\\delta_{TD}$')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.subplot(2,1,2)
plt.plot(t_arr, V_hist, color='g')
plt.xlabel('Time step')
plt.ylabel('V')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.tight_layout()

# ---------------------------
# Plot results
# ---------------------------
fig = plt.figure(figsize=(12,6))
plt.suptitle('RL-PID (paper-style) Gains')


plt.subplot(3,1,1)
plt.plot(t_arr, KP_hist, color='b')
plt.xlabel('Time step')
plt.ylabel('$K_p$')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.subplot(3,1,2)
plt.plot(t_arr, KI_hist, color='g')
plt.xlabel('Time step')
plt.ylabel('$K_i$')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.subplot(3,1,3)
plt.plot(t_arr, KD_hist, color='m')
plt.xlabel('Time step')
plt.ylabel('$K_d$')
plt.xlim([0,max(t_arr)])
plt.grid(color = 'gray', linestyle = ':')

plt.tight_layout()
