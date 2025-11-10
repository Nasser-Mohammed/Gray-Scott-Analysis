import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ======================================================
# Parameters
# ======================================================
n = 256
Du, Dv = 0.1175, 0.08
F, k = 0.037, 0.06
dt = 0.2
steps = 100000
steady_tol = 1e-6      # threshold for "no change"
steady_frames = 8       # must remain steady for N checks

# ======================================================
# Initialization
# ======================================================
u = np.ones((n, n))
v = np.zeros((n, n))

# Circular blob (JS-style initialization)
r = 20
cx, cy = n // 2, n // 2
y, x = np.ogrid[:n, :n]
mask = (x - cx)**2 + (y - cy)**2 <= r**2
v[mask] = 1.0

# ======================================================
# 8-neighbor Laplacian (periodic BCs)
# ======================================================
@njit
def laplacian(Z):
    n, m = Z.shape
    L = np.empty_like(Z)
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        for j in range(m):
            jp = (j + 1) % m
            jm = (j - 1) % m
            L[i, j] = (
                Z[ip, j] + Z[im, j] + Z[i, jp] + Z[i, jm] +
                Z[ip, jp] + Z[ip, jm] + Z[im, jp] + Z[im, jm]
                - 8.0 * Z[i, j]
            )
    return L

# ======================================================
# Gray–Scott update step
# ======================================================
@njit
def step(u, v, Du, Dv, F, k, dt):
    Lu = laplacian(u)
    Lv = laplacian(v)
    uvv = u * v * v
    du = Du * Lu - uvv + F * (1 - u)
    dv = Dv * Lv + uvv - (F + k) * v
    u += dt * du
    v += dt * dv
    return u, v, du, dv

# ======================================================
# Visualization setup
# ======================================================
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((n, n, 3), dtype=np.uint8),
               origin="lower", interpolation="none")
ax.set_axis_off()
title = ax.set_title("Gray–Scott model (steady-state detection)")

# ======================================================
# Simulation loop
# ======================================================
steady_counter = 0
for i in range(steps):
    if not plt.fignum_exists(fig.number):
        break

    # Perform several updates between frames
    for _ in range(5):
        u_old, v_old = u.copy(), v.copy()
        u, v, du, dv = step(u, v, Du, Dv, F, k, dt)

    # --- detect convergence ---
    delta_u = np.max(np.abs(u - u_old))
    delta_v = np.max(np.abs(v - v_old))
    delta_total = delta_u + delta_v

    if delta_total < steady_tol:
        steady_counter += 1
    else:
        steady_counter = 0

    if steady_counter >= steady_frames:
        print(f"Steady state reached at t = {i * dt:.2f}, Δ = {delta_total:.2e}")
        break

    # --- display every 50 frames ---
    if i % 50 == 0:
        u_clamp = np.clip(u, 0, 1)
        v_clamp = np.clip(v, 0, 1)
        r = np.maximum(u_clamp - v_clamp, 0) * 255
        g = v_clamp * 255
        b = np.maximum(1 - u_clamp, 0) * 255
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

        im.set_data(rgb)
        title.set_text(f"Gray–Scott model | t = {i * dt:.1f} | Δ={delta_total:.2e}")
        plt.pause(0.05)

plt.ioff()
plt.show()
