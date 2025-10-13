# RL-PID following Guan & Yamamoto (explicit RBF actor + RBF critic)
# Pure numpy implementation (no autograd) implementing eqs (19),(20),(21) and (23)-(25).
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

np.random.seed(0)

# ---------------------------
# Plant (same nonlinear plant you used, but with small noise)
# ---------------------------
def plant_step(y_prev, u, y_pprev, noise_std=0.01):
    noise = np.random.normal(0, noise_std)
    y = ((y_prev * y_pprev * (y_prev + 2.5)) / (1 + y_prev**2 + y_pprev**2)) + u + noise
    return float(y)

# ---------------------------
# Desired response (piecewise)
# ---------------------------
def desired_response(t):
    if t < 100:
        return 2.5
    elif t < 200:
        return 5.0
    elif t < 300:
        return 4.0
    else:
        return 3.0
    # return math.sin(0.01*t)
    # if t < 100:
    #     return 2.5
    # elif t < 200:
    #     return 5.0
    # elif t < 300:
    #     return 4.0
    # elif t < 400:
    #     return 3.0
    # elif t < 500:
    #     return 2.5
    # elif t < 600:
    #     return 5.0
    # elif t < 700:
    #     return 4.0
    # else:
    #     return 3.0

# ---------------------------
# RBF utilities
# ---------------------------
def rbf_activations(x, centers, sigmas):
    # x: (D,)
    # centers: (M, D)
    # sigmas: (M,) standard deviation per center
    diff = centers - x[None, :]  # (M, D)
    sq = np.sum(diff**2, axis=1)  # (M,)
    # avoid division by zero
    s = np.maximum(sigmas, 1e-8)
    phi = np.exp(-sq / (2.0 * s**2))
    return phi  # shape (M,)

# ---------------------------
# Algorithm hyperparameters
# ---------------------------
T = 400                        # time steps
gamma = 0.98                   # discount
noise_std = 0.01               # plant noise std

# RBF sizes
M_actor = 16                   # number of RBF units for actor
M_critic = 16                  # number of RBF units for critic
state_dim_actor = 3            # e, de, dde (or whichever features you choose)
state_dim_critic = 3

# Learning rates (conservative)
alpha_v = 1e-4
alpha_mu = 1e-4
alpha_sigma = 1e-4
alpha_w = 1e-4   # actor weights learning rate (applies to P/I/D)
# you can use separate rates for P/I/D if needed

# Prior knowledge: sign of dy/du (1 or -1). Paper uses this to ensure correct direction.
jac_sign = 1.0  # set to -1.0 if the system is reverse-acting

# ---------------------------
# Initialize RBF centers and widths (simple grid over expected input space)
# We'll normalize actor/critic inputs roughly to [-1,1] scale in usage
# ---------------------------
def make_centers(M, dim, spread=1.0):
    # place centers randomly in [-spread, spread]^dim
    return np.random.uniform(-spread, spread, size=(M, dim))

centers_actor = make_centers(M_actor, state_dim_actor, spread=1.0)
sigmas_actor = np.ones(M_actor) * 0.6

centers_critic = make_centers(M_critic, state_dim_critic, spread=1.0)
sigmas_critic = np.ones(M_critic) * 0.6

# ---------------------------
# Weights init
# ---------------------------
wP = np.random.randn(M_actor) * 1e-3
wI = np.random.randn(M_actor) * 1e-3
wD = np.random.randn(M_actor) * 1e-3

v = np.random.randn(M_critic) * 1e-3

# ---------------------------
# Helper: actor outputs KP,KI,KD from actor features
# ---------------------------
def actor_gains_from_features(phi):
    KP = np.dot(wP, phi)
    KI = np.dot(wI, phi)
    KD = np.dot(wD, phi)
    return float(KP), float(KI), float(KD)

# ---------------------------
# Storage
# ---------------------------
y_hist = []
yd_hist = []
u_hist = []
KP_hist = []
KI_hist = []
KD_hist = []
td_hist = []
V_hist = []

# ---------------------------
# Initial states
# ---------------------------
y_pprev = 0.0
y_prev = 0.0
y = 0.0
u_prev = 0.0

# initial critic value V(t)
# compute initial state features for critic (we will use psi = [e, dy, ddy] style)
e_init = desired_response(0) - y
dy_init = y - y_prev
ddy_init = y - 2*y_prev + y_pprev # rough initial
psi_critic = np.array([e_init, -dy_init, -ddy_init])  # paper often uses negatives for derivatives
phi_c = rbf_activations(psi_critic, centers_critic, sigmas_critic)
V_current = float(np.dot(v, phi_c))

# ---------------------------
# Main loop (follows paper order):
# 1) compute actor gains from actor features at time t (use psi_actor)
# 2) compute control increment u_inc (velocity form)
# 3) apply control to plant to get y(t+1)
# 4) compute reward r(t)
# 5) compute critic features phi(t+1) and V(t+1)
# 6) compute TD error delta = r + gamma*V_next - V_current
# 7) update critic (v, mu, sigma) per eq (23-25)
# 8) update actor weights using explicit gradients from (19-21) using jac_sign
# 9) shift indices
# ---------------------------
num_epochs = 100
for t in tqdm(range(T)):
    # desired
    yd = desired_response(t)
    # current error and derivatives (computed from available y_prev/y_pprev)
    e = yd - y
    dy = y - y_prev
    ddy = y - 2*y_prev + y_pprev  # simple placeholder
    # many papers use psi = [e, -dy, -ddy] as input; choose consistently:
    psi_actor = np.array([e, -dy, -ddy])

    for epoch in range(num_epochs):
      # normalize psi_actor if desired (here kept raw but center spread expects ~[-1,1])
      # compute actor features
      phi_actor = rbf_activations(psi_actor, centers_actor, sigmas_actor)  # shape (M_actor,)

      # actor outputs (gains)
      KP, KI, KD = actor_gains_from_features(phi_actor)
      # optional: bound gains to reasonable ranges
      KP = np.clip(KP, -50.0, 50.0)
      KI = np.clip(KI, -10.0, 10.0)
      KD = np.clip(KD, -10.0, 10.0)

      # compute incremental PID control (velocity form)
      # paper uses: u(t) = u(t-1) + K_I(t)*e(t) - K_P(t)*Delta y(t) - K_D(t)*Delta^2 y(t)
      delta_y = y - y_prev
      delta2_y = y - 2*y_prev + y_pprev  # small placeholder (requires earlier values, we use approximate)
      # Use the velocity-update form exactly as paper:
      u_inc = KI * e - KP * delta_y - KD * delta2_y
      u = u_prev + u_inc

      # apply plant
      y_next = plant_step(y, u, y_prev, noise_std=noise_std)

      # reward using next output as in paper's derivation
      r = 0.5 * (yd - y_next)**2

      # critic features at time t+1 (use psi_critic = [e_next, -dy_next, -ddy_next])
      e_next = yd - y
      dy_next = y_next - y
      ddy_next = y_next - 2*y + y_prev
      psi_critic_next = np.array([e_next, -dy_next, -ddy_next])
      phi_critic_next = rbf_activations(psi_critic_next, centers_critic, sigmas_critic)
      V_next = float(np.dot(v, phi_critic_next))

      # compute TD error using V_current from previous iteration
      delta_TD = r + gamma * V_next - V_current

      # ------------------ Critic updates (eq 23-25) ------------------
      # output weights v_j(t+1) = v_j(t) + alpha_v * delta_TD * Phi_j(t)
      phi_critic_current = rbf_activations(np.array([e, -dy, -ddy]), centers_critic, sigmas_critic)
      v += alpha_v * delta_TD * phi_critic_current

      # centers and sigmas update (eq 24 and 25)
      # mu_ij(t+1) = mu_ij + alpha_mu * delta_TD * v_j * Phi_j * (psi_i - mu_ij)/sigma_j^2
      # sigma_j(t+1) = sigma_j + alpha_sigma * delta_TD * v_j * Phi_j * ||psi - mu||^2 / sigma_j^3

      psi_for_update = np.array([e, -dy, -ddy])  # use current psi for critic updates
      for j in range(M_critic):
          phi_j = phi_critic_current[j]
          vj = v[j]
          # vector difference
          diff = psi_for_update - centers_critic[j]
          dist2 = np.sum(diff**2)
          sig = sigmas_critic[j]
          if sig < 1e-6:
              sig = 1e-6
          # center update
          centers_critic[j] = centers_critic[j] + alpha_mu * delta_TD * vj * phi_j * diff / (sig**2)
          # sigma update
          sigmas_critic[j] = sigmas_critic[j] + alpha_sigma * delta_TD * vj * phi_j * (dist2) / (sig**3)
          # keep sigma positive
          sigmas_critic[j] = max(sigmas_critic[j], 1e-4)

      # ------------------ Actor updates (eq 19-21 with jac_sign substitution) ------------------
      # Use the fully explicit derivatives (we derived earlier):
      # For P weights:
      # dJ/dwP_j = - delta_TD * (y_next - yd) * (y_prev - y_pprev) * Phi_actor_j * jac_sign
      # For I weights:
      # dJ/dwI_j = - delta_TD * (y_next - yd) * e * Phi_actor_j * jac_sign
      # For D weights:
      # dJ/dwD_j = - delta_TD * (y_next - yd) * (y_prev - 2*y_pprev + 0.0) * Phi_actor_j * jac_sign
      # (These follow from the explicit chain-rule we wrote earlier; signs chosen to match gradient descent)

      # note: including (y_next - yd) factor (reward derivative) gives explicit dependence and is safe here
      factor = abs((yd - y_next))*np.sign((yd - y_next))   # this multiplies the rest
      # factor = np.sign((y_next - y_prev) / (u - u_prev)) if abs(u - u_prev) > 1e-6 else jac_sign
      # factor = -abs((y_next - yd)/ (u - u_prev))*np.sign((y_next - yd)/(u - u_prev)) if abs(u - u_prev) > 1e-6 else -abs((y_next - yd))*np.sign((y_next - yd))

      # factor = abs((y_next - y) / (u - u_prev))* np.sign((y_next - y) / (u - u_prev)) if abs(u - u_prev) > 1e-6 else abs((y_next - y))* np.sign((y_next - y))
      # delta_TD multiplies gradient (and gradient descent uses subtract alpha * grad)
      for j in range(M_actor):
          phi_j = phi_actor[j]
          # common factor for gradient
          grad_wP_j = - delta_TD * factor * (y - y_prev) * phi_j
          grad_wI_j = - delta_TD * factor * e * phi_j
          grad_wD_j = - delta_TD * factor * (y - 2*y_prev + y_pprev) * phi_j

          # gradient descent step: w <- w - alpha * dJ/dw
          wP[j] = wP[j] - alpha_w * grad_wP_j
          wI[j] = wI[j] - alpha_w * grad_wI_j
          wD[j] = wD[j] - alpha_w * grad_wD_j

    # ------------------ shift time indices for next iteration ------------------
    u_prev = u
    y_pprev, y_prev, y = y_prev, y, y_next
    V_current = V_next

    # store
    y_hist.append(y_next)
    yd_hist.append(yd)
    u_hist.append(u_prev)
    KP_hist.append(KP)
    KI_hist.append(KI)
    KD_hist.append(KD)
    td_hist.append(delta_TD)
    V_hist.append(V_current)


# ---------------------------
# Plot results
# ---------------------------
t_arr = np.arange(len(y_hist))
plt.figure(figsize=(12,6))
plt.plot(t_arr, y_hist, label='Output (y)', color='k')
plt.plot(t_arr, yd_hist, '--', label='Desired (yd)', color='r')
plt.xlabel('Time step')
plt.ylabel('y')
plt.title('RL-PID (paper-style) Output vs Desired')
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
