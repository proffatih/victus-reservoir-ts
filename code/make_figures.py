"""Generate all publication figures (vector PDF + 300 dpi PNG).
Colorblind-safe palette (Wong 2011). Run after run_experiments.py."""
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

# Wong colorblind-safe palette
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442",
      "sky": "#56B4E9", "black": "#000000"}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "lines.linewidth": 1.8,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", name)


# --- Fig 1: Lorenz attractor + prediction overlay ---------------------
def fig_attractor():
    d = np.load(os.path.join(RES, "lorenz_pred.npz"))
    true, pred = d["true"], d["pred"]
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    n = min(2000, len(true))
    ax1.plot(true[:n, 0], true[:n, 1], true[:n, 2], color=CB["blue"],
             lw=0.5, alpha=0.8)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title("(a) Lorenz attractor (ground truth)")

    ax2 = fig.add_subplot(1, 2, 2)
    dt = float(d["dt"]); t = np.arange(len(true)) * dt
    m = min(600, len(true))
    ax2.plot(t[:m], true[:m, 0], color=CB["black"], lw=1.6, label="True")
    ax2.plot(t[:m], pred[:m, 0], color=CB["orange"], lw=1.4, ls="--",
             label="ESN free-run")
    ax2.set_xlabel("Time  $t$  (model time units)")
    ax2.set_ylabel("$x(t)$")
    ax2.set_title("(b) Autonomous prediction of $x$-coordinate")
    ax2.legend(loc="upper right")
    save(fig, "fig1_attractor_prediction")


# --- Fig 2: Mackey-Glass prediction overlay ---------------------------
def fig_mg_pred():
    d = np.load(os.path.join(RES, "mackeyglass_pred.npz"))
    true, pred = d["true"].ravel(), d["pred"].ravel()
    lt = json.load(open(os.path.join(RES, "summary.json")))["lyapunov_time_steps"]["mackey_glass"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    m = min(800, len(true))
    t = np.arange(m)
    ax.plot(t, true[:m], color=CB["black"], lw=1.6, label="True")
    ax.plot(t, pred[:m], color=CB["red"], lw=1.4, ls="--", label="ESN free-run")
    # mark Lyapunov-time gridlines
    for k in range(1, int(m / lt) + 1):
        ax.axvline(k * lt, color=CB["sky"], ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Prediction step  $n$  (dotted lines: Lyapunov times $\\Lambda$)")
    ax.set_ylabel("$x(n)$")
    ax.set_title("Mackey--Glass ($\\tau=17$) autonomous prediction")
    ax.legend(loc="upper right")
    save(fig, "fig2_mackeyglass_prediction")


# --- Fig 3: NRMSE & VPT vs spectral radius ----------------------------
def fig_spectral():
    df = pd.read_csv(os.path.join(RES, "sweep_spectral_radius.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].errorbar(df.spectral_radius, df.nrmse, yerr=df.nrmse_std,
                   color=CB["blue"], marker="o", ms=4, capsize=2)
    ax[0].axvline(1.0, color=CB["red"], ls="--", lw=1, label="$\\rho=1$ (edge of stability)")
    ax[0].set_xlabel("Spectral radius  $\\rho$")
    ax[0].set_ylabel("NRMSE (1 Lyapunov time)")
    ax[0].set_yscale("log"); ax[0].set_title("(a)"); ax[0].legend()
    ax[1].errorbar(df.spectral_radius, df.vpt_lyap, yerr=lyap_err(df),
                   color=CB["green"], marker="s", ms=4, capsize=2)
    ax[1].axvline(1.0, color=CB["red"], ls="--", lw=1)
    ax[1].set_xlabel("Spectral radius  $\\rho$")
    ax[1].set_ylabel("Valid time  (Lyapunov times $\\Lambda$)")
    ax[1].set_title("(b)")
    save(fig, "fig3_spectral_radius")


# --- Fig 4: heatmap rho x leak ----------------------------------------
def fig_heatmap():
    d = np.load(os.path.join(RES, "heatmap_rho_leak.npz"))
    rhos, leaks, H, Hv = d["rhos"], d["leaks"], d["nrmse"], d["vpt"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    Hp = np.log10(np.clip(H, 1e-4, None))
    im0 = ax[0].imshow(Hp, origin="lower", aspect="auto", cmap="viridis",
                       extent=[rhos.min(), rhos.max(), leaks.min(), leaks.max()])
    ax[0].set_xlabel("Spectral radius  $\\rho$"); ax[0].set_ylabel("Leak rate  $\\alpha$")
    ax[0].set_title("(a) $\\log_{10}$ NRMSE")
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(Hv, origin="lower", aspect="auto", cmap="magma",
                       extent=[rhos.min(), rhos.max(), leaks.min(), leaks.max()])
    ax[1].set_xlabel("Spectral radius  $\\rho$"); ax[1].set_ylabel("Leak rate  $\\alpha$")
    ax[1].set_title("(b) Valid prediction time (steps)")
    fig.colorbar(im1, ax=ax[1])
    save(fig, "fig4_heatmap_rho_leak")


# --- Fig 5: reservoir size + leak + connectivity ----------------------
def fig_size_leak_conn():
    sz = pd.read_csv(os.path.join(RES, "sweep_reservoir_size.csv"))
    lk = pd.read_csv(os.path.join(RES, "sweep_leak_rate.csv"))
    cn = pd.read_csv(os.path.join(RES, "sweep_connectivity.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].errorbar(sz.n_reservoir, sz.vpt_lyap, yerr=lyap_err(sz),
                   color=CB["blue"], marker="o", ms=4, capsize=2)
    ax[0].set_xlabel("Reservoir size  $N$"); ax[0].set_ylabel("Valid time ($\\Lambda$)")
    ax[0].set_title("(a) Reservoir size")
    ax[1].errorbar(lk.leak_rate, lk.vpt_lyap, yerr=lyap_err(lk),
                   color=CB["orange"], marker="s", ms=4, capsize=2)
    ax[1].set_xlabel("Leak rate  $\\alpha$"); ax[1].set_ylabel("Valid time ($\\Lambda$)")
    ax[1].set_title("(b) Leak rate")
    ax[2].errorbar(cn.density, cn.vpt_lyap, yerr=lyap_err(cn),
                   color=CB["green"], marker="^", ms=5, capsize=2)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Connection density"); ax[2].set_ylabel("Valid time ($\\Lambda$)")
    ax[2].set_title("(c) Connectivity")
    save(fig, "fig5_size_leak_connectivity")


def lyap_err(df):
    """Convert the VPT s.d. from steps to Lyapunov-time units."""
    # steps-per-Lyapunov-time = vpt_steps / vpt_lyap (constant across rows)
    steps_per_lyap = (df.vpt_steps / df.vpt_lyap.replace(0, np.nan)).median()
    return (df.vpt_std / steps_per_lyap).values


# --- Fig 6: baseline comparison bar -----------------------------------
def fig_baselines():
    df = pd.read_csv(os.path.join(RES, "baselines.csv"))
    tasks = df.task.unique()
    models = ["ESN", "Linear AR", "MLP", "ARIMA"]
    colors = {"ESN": CB["blue"], "Linear AR": CB["orange"],
              "MLP": CB["green"], "ARIMA": CB["red"]}
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(tasks)); w = 0.2
    for i, m in enumerate(models):
        vals = []
        for t in tasks:
            sub = df[(df.task == t) & (df.model == m)]
            vals.append(sub.nrmse.values[0] if len(sub) else np.nan)
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[m])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=8)
    ax.set_ylabel("NRMSE (log scale)")
    ax.set_title("ESN vs. baselines across benchmark tasks")
    ax.legend(ncol=2)
    save(fig, "fig6_baselines")


# --- Fig 7: memory capacity -------------------------------------------
def fig_memory():
    d = np.load(os.path.join(RES, "memory_capacity.npz"))
    delays = d["delays"]
    keys = [k for k in d.files if k.startswith("rho_")]
    order = sorted(keys, key=lambda s: float(s.split("_")[1]))
    cols = [CB["blue"], CB["green"], CB["orange"], CB["red"]]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    totals = []
    for c, k in zip(cols, order):
        rho = k.split("_")[1]
        ax[0].plot(delays, d[k], color=c, label=f"$\\rho={rho}$")
        totals.append((float(rho), d[k].sum()))
    ax[0].set_xlabel("Delay  $k$"); ax[0].set_ylabel("Forgetting curve  $r^2(k)$")
    ax[0].set_title("(a) Memory function"); ax[0].legend()
    tr = np.array(totals)
    ax[1].plot(tr[:, 0], tr[:, 1], color=CB["purple"], marker="o", ms=6)
    ax[1].set_xlabel("Spectral radius  $\\rho$")
    ax[1].set_ylabel("Total memory capacity  MC")
    ax[1].set_title("(b) Capacity vs. $\\rho$")
    save(fig, "fig7_memory_capacity")


if __name__ == "__main__":
    fig_attractor()
    fig_mg_pred()
    fig_spectral()
    fig_heatmap()
    fig_size_leak_conn()
    fig_baselines()
    fig_memory()
    print("ALL FIGURES DONE")
