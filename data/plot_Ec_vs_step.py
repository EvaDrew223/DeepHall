import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'

N = 9
Q = 15/2
nu = 1/2
N_layer = 4
kappa = 1

dict_path = Path(__file__).parent / f"seed_1758632847/tillicum/filling_1over2/DeepHall_n{N}l{int(2*Q)}_NN{N_layer}L_3360"
csv_path = dict_path / "train_stats.csv"
df = pd.read_csv(csv_path)

# Ensure numeric types
steps = df["step"].astype(int).to_numpy()
energy = df["energy"].astype(float).to_numpy()

window = 500
smoothed = np.empty_like(energy, dtype=float)

for i in range(len(energy)):
    start = max(0, i - window + 1)
    w = energy[start:i + 1]
    # IQR-based outlier removal within the window
    q1 = np.nanpercentile(w, 25)
    q3 = np.nanpercentile(w, 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    w_filtered = w[(w >= low) & (w <= high)]
    if w_filtered.size == 0:
        w_filtered = w  # fallback if all were filtered
    smoothed[i] = np.nanmean(w_filtered)

# %%
smoothed_plot = (smoothed - N**2/2/np.sqrt(Q) - N/2/kappa) * np.sqrt(2*Q*nu/N) / N # N/2/kappa

plt.figure(figsize=(8, 7))
# plt.plot(steps, energy, color="lightgray", linewidth=1, label="raw energy")
plt.plot(steps, smoothed_plot, color="C0", linewidth=2, label=f"moving avg (window={window}, IQR outliers removed)")
plt.xlim(-500, 5000)#200000)
# plt.ylim(-0.4115-0.0004, -0.4095+0.0001)
# plt.ylim(-0.434-0.0005, -0.426 +0.001)
# plt.ylim(-0.2-0.002, -0.18)
ax = plt.gca()
# major_ticks_y = np.arange(-0.4115, -0.4095 + 1e-12, 0.0005)
# minor_ticks_y = np.arange(-0.4115-0.0004, -0.4095 + 1e-12, 0.0001)
# major_ticks_y = np.arange(-0.434, -0.426 + 1e-12, 0.002)
# minor_ticks_y = np.arange(-0.434-0.001, -0.426 +0.001 + 1e-12, 0.0005)
# ax.set_yticks(major_ticks_y)
# ax.tick_params(axis='y', which='major', width=1.5, length=5)
# ax.set_yticks(minor_ticks_y, minor=True)
# major_ticks_x = np.arange(0, 200000+1, 100000)
# minor_ticks_x = np.arange(0, 200000+ 1e-12, 20000)
# ax.set_xticks(major_ticks_x)
# ax.tick_params(axis='x', which='major', width=1.5, length=5)
# ax.set_xticks(minor_ticks_x, minor=True)
ax.tick_params(axis='both', which='both', labelsize=18)
plt.xlabel("training step", fontsize=18)
plt.ylabel(r"$E_c/N\hbar\omega_c \kappa$", fontsize=18)
plt.title(f"NN layers = {N_layer}", fontsize=18)
plt.legend([r"$\nu=1/2$"], fontsize=18)
plt.tight_layout()
plt.savefig(str(dict_path / "Ec_vs_step.png"), dpi=300)
# %%
