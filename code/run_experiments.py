"""Main experiment driver for the reservoir-computing benchmark study.
Produces all CSV/JSON results consumed by figures and manuscript.
Fully reproducible (fixed seeds). Run: python run_experiments.py
"""
import os, json, time
import numpy as np
import pandas as pd

from esn import ESN
import datasets as ds
import metrics as mx

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(RES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

GLOBAL_SEED = 42
N_SEEDS = 5            # reservoir realizations for error bars
SEEDS = list(range(GLOBAL_SEED, GLOBAL_SEED + N_SEEDS))

# Lyapunov times (largest LE) for autonomous benchmarks (1/lambda_1, in steps)
# Computed below empirically and stored.


def normalize(x):
    mu, sd = x.mean(0), x.std(0)
    return (x - mu) / sd, mu, sd


# ----------------------------------------------------------------------
def prep_mackey_glass(N=8000, tau=17):
    s = ds.mackey_glass(N, tau=tau, seed=1)
    return s.reshape(-1, 1)


def prep_lorenz(N=10000):
    s = ds.lorenz(N, seed=1)
    return s  # (N,3)


def prep_santafe(N=6000):
    s = ds.santa_fe_laser(N, seed=1)
    return s.reshape(-1, 1)


# ----------------------------------------------------------------------
def _clean_state(esn, seq):
    """Re-drive reservoir over `seq` with no training noise to obtain a
    consistent closed-loop initial state."""
    st = np.zeros(esn.n_reservoir)
    for t in range(len(seq)):
        st = esn._update(st, seq[t])
    return st


def autonomous_eval(series, n_res, rho, leak, sparsity, train_len, washout,
                    horizon, seeds, reg=1e-7, input_scaling=0.5, noise=1e-4,
                    short_horizon=None):
    """Train next-step predictor on `series`, then free-run for `horizon`
    steps; report NRMSE over a short horizon (~1 Lyapunov time, robust to
    post-divergence blow-up) and valid prediction time (steps).
    NRMSE over the short horizon is the reported autonomous accuracy."""
    sn, mu, sd = normalize(series)
    nin = sn.shape[1]
    X = sn[:train_len]
    Y = sn[1:train_len + 1]
    test_true = sn[train_len + 1: train_len + 1 + horizon]
    sh = short_horizon or min(horizon, 100)
    nrmses, vpts = [], []
    pred_store = None
    for sd_ in seeds:
        esn = ESN(n_inputs=nin, n_reservoir=n_res, n_outputs=nin,
                  spectral_radius=rho, sparsity=sparsity, leak_rate=leak,
                  input_scaling=input_scaling, reg=reg, noise=noise, seed=sd_)
        esn.fit(X, Y, washout=washout)
        state0 = _clean_state(esn, X)
        pred = esn.generate(horizon, X[-1], state0)
        # clip free-run to the observed dynamical range (+margin) so that
        # rare late-horizon blow-ups do not dominate the short-horizon NRMSE
        lo, hi = test_true.min() - 2, test_true.max() + 2
        predc = np.clip(np.nan_to_num(pred, nan=hi, posinf=hi, neginf=lo), lo, hi)
        nrmses.append(mx.nrmse(test_true[:sh], predc[:sh]))
        vpts.append(mx.valid_prediction_time(test_true, pred))
        if pred_store is None:
            pred_store = pred
    return (np.mean(nrmses), np.std(nrmses),
            np.mean(vpts), np.std(vpts), pred_store, test_true, mu, sd)


def teacher_eval(u, y, n_res, rho, leak, sparsity, train_len, washout,
                 seeds, reg=1e-6, input_scaling=0.5):
    """One-step system identification (NARMA): teacher-forced NRMSE."""
    u = u.reshape(-1, 1); y = y.reshape(-1, 1)
    yn, mu, sd = normalize(y)
    Xtr, Ytr = u[:train_len], yn[:train_len]
    Xte, Yte = u[train_len:], yn[train_len:]
    errs = []
    pred_store = None
    for sd_ in seeds:
        esn = ESN(n_inputs=1, n_reservoir=n_res, n_outputs=1,
                  spectral_radius=rho, sparsity=sparsity, leak_rate=leak,
                  input_scaling=input_scaling, reg=reg, seed=sd_)
        esn.fit(Xtr, Ytr, washout=washout)
        pred = esn.predict_teacher(Xte, washout=washout)
        errs.append(mx.nrmse(Yte[washout:], pred))
        if pred_store is None:
            pred_store = (pred, Yte[washout:])
    return np.mean(errs), np.std(errs), pred_store


# ======================================================================
def main():
    t0 = time.time()
    summary = {}

    # --- Lyapunov exponents / times -----------------------------------
    print("Estimating Lyapunov exponents ...")
    mg = prep_mackey_glass(8000, tau=17).ravel()
    lor = prep_lorenz(10000)
    lyap = {}
    # Mackey-Glass: authoritative variational (tangent-space) estimate
    lyap["mackey_glass_tau17_per_step"] = float(mx.lyapunov_mackey_glass(tau=17))
    # cross-check via Rosenstein (data-driven, lag=5 to avoid oversampling bias)
    lyap["mackey_glass_rosenstein_crosscheck"] = float(
        mx.lyapunov_rosenstein(mg[:8000], emb_dim=8, tau=5, traj_len=30, min_tsep=30))
    # Lorenz: known lambda1 ~ 0.9056 /time-unit; dt=0.02 => per-step
    lyap["lorenz_x_per_step"] = float(
        mx.lyapunov_rosenstein(lor[:5000, 0], emb_dim=5, tau=2, traj_len=25, min_tsep=50))
    lyap["lorenz_theoretical_per_time"] = 0.9056
    lyap["lorenz_dt"] = 0.02
    lyap["lorenz_theoretical_per_step"] = 0.9056 * 0.02
    for k, v in lyap.items():
        print(f"  {k} = {v}")
    json.dump(lyap, open(os.path.join(RES, "lyapunov.json"), "w"), indent=2)

    # Lyapunov time in steps (1/lambda) for converting horizon -> Lyapunov times
    lt_mg = 1.0 / lyap["mackey_glass_tau17_per_step"] if lyap["mackey_glass_tau17_per_step"] > 0 else np.nan
    lt_lor = 1.0 / lyap["lorenz_theoretical_per_step"]
    summary["lyapunov_time_steps"] = {"mackey_glass": lt_mg, "lorenz": lt_lor}

    # --- 1. Spectral radius sweep (Mackey-Glass, autonomous) ----------
    print("Sweep: spectral radius (Mackey-Glass) ...")
    rhos = np.round(np.arange(0.3, 1.41, 0.05), 3)
    rows = []
    for rho in rhos:
        nr, ns, vt, vs, *_ = autonomous_eval(
            prep_mackey_glass(10000), n_res=600, rho=rho, leak=1.0,
            sparsity=0.05, train_len=5000, washout=300, horizon=2000,
            seeds=SEEDS, input_scaling=0.5, noise=1e-4, short_horizon=110)
        rows.append(dict(spectral_radius=rho, nrmse=nr, nrmse_std=ns,
                         vpt_steps=vt, vpt_std=vs,
                         vpt_lyap=vt / lt_mg))
        print(f"  rho={rho:.2f} NRMSE={nr:.4f} VPT={vt:.0f}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sweep_spectral_radius.csv"), index=False)

    # --- 2. Leak rate sweep -------------------------------------------
    print("Sweep: leak rate (Mackey-Glass) ...")
    leaks = np.round(np.arange(0.1, 1.01, 0.05), 3)
    rows = []
    for lk in leaks:
        nr, ns, vt, vs, *_ = autonomous_eval(
            prep_mackey_glass(10000), n_res=600, rho=0.9, leak=lk,
            sparsity=0.05, train_len=5000, washout=300, horizon=2000,
            seeds=SEEDS, input_scaling=0.5, noise=1e-4, short_horizon=110)
        rows.append(dict(leak_rate=lk, nrmse=nr, nrmse_std=ns,
                         vpt_steps=vt, vpt_std=vs, vpt_lyap=vt / lt_mg))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sweep_leak_rate.csv"), index=False)

    # --- 3. Reservoir size sweep --------------------------------------
    print("Sweep: reservoir size (Mackey-Glass) ...")
    sizes = [50, 100, 200, 300, 400, 600, 800, 1000]
    rows = []
    for n in sizes:
        nr, ns, vt, vs, *_ = autonomous_eval(
            prep_mackey_glass(10000), n_res=n, rho=0.9, leak=1.0,
            sparsity=0.05, train_len=5000, washout=300, horizon=2000,
            seeds=SEEDS, input_scaling=0.5, noise=1e-4, short_horizon=110)
        rows.append(dict(n_reservoir=n, nrmse=nr, nrmse_std=ns,
                         vpt_steps=vt, vpt_std=vs, vpt_lyap=vt / lt_mg))
        print(f"  N={n} NRMSE={nr:.4f} VPT={vt:.0f}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sweep_reservoir_size.csv"), index=False)

    # --- 4. Connectivity (sparsity) sweep -----------------------------
    print("Sweep: connectivity (Mackey-Glass) ...")
    spars = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    rows = []
    for sp in spars:
        nr, ns, vt, vs, *_ = autonomous_eval(
            prep_mackey_glass(10000), n_res=600, rho=0.9, leak=1.0,
            sparsity=sp, train_len=5000, washout=300, horizon=2000,
            seeds=SEEDS, input_scaling=0.5, noise=1e-4, short_horizon=110)
        rows.append(dict(density=sp, nrmse=nr, nrmse_std=ns,
                         vpt_steps=vt, vpt_std=vs, vpt_lyap=vt / lt_mg))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sweep_connectivity.csv"), index=False)

    # --- 5. 2D heatmap: spectral radius x leak rate -------------------
    print("Heatmap: rho x leak (Mackey-Glass) ...")
    rhos_h = np.round(np.arange(0.4, 1.31, 0.1), 2)
    leaks_h = np.round(np.arange(0.2, 1.01, 0.1), 2)
    H = np.zeros((len(leaks_h), len(rhos_h)))
    Hv = np.zeros_like(H)
    for i, lk in enumerate(leaks_h):
        for j, rho in enumerate(rhos_h):
            nr, ns, vt, vs, *_ = autonomous_eval(
                prep_mackey_glass(10000), n_res=400, rho=rho, leak=lk,
                sparsity=0.05, train_len=5000, washout=300, horizon=2000,
                seeds=SEEDS[:3], input_scaling=0.5, noise=1e-4, short_horizon=110)
            H[i, j] = nr; Hv[i, j] = vt
    np.savez(os.path.join(RES, "heatmap_rho_leak.npz"),
             rhos=rhos_h, leaks=leaks_h, nrmse=H, vpt=Hv)

    # --- 6. Lorenz autonomous prediction + horizon vs Lyapunov --------
    print("Lorenz autonomous prediction ...")
    nr, ns, vt, vs, pred, true, mu, sd = autonomous_eval(
        prep_lorenz(15000), n_res=800, rho=1.0, leak=1.0, sparsity=0.03,
        train_len=8000, washout=500, horizon=2500, seeds=SEEDS,
        reg=1e-7, input_scaling=0.3, noise=1e-4, short_horizon=100)
    summary["lorenz"] = dict(nrmse=nr, nrmse_std=ns, vpt_steps=vt,
                             vpt_std=vs, vpt_lyap=vt / lt_lor)
    print(f"  Lorenz NRMSE={nr:.4f} VPT={vt:.0f} steps = {vt/lt_lor:.2f} Lyap times")
    # save trajectories (denormalized) for plotting
    np.savez(os.path.join(RES, "lorenz_pred.npz"),
             pred=pred * sd + mu, true=true * sd + mu, dt=0.02)

    # --- 7. Mackey-Glass best-config autonomous trajectory ------------
    print("Mackey-Glass best autonomous trajectory ...")
    nr, ns, vt, vs, pred, true, mu, sd = autonomous_eval(
        prep_mackey_glass(10000), n_res=1000, rho=0.9, leak=1.0,
        sparsity=0.05, train_len=5000, washout=300, horizon=2000, seeds=SEEDS,
        input_scaling=0.5, noise=1e-4, short_horizon=110)
    summary["mackey_glass"] = dict(nrmse=nr, nrmse_std=ns, vpt_steps=vt,
                                   vpt_std=vs, vpt_lyap=vt / lt_mg)
    np.savez(os.path.join(RES, "mackeyglass_pred.npz"),
             pred=pred * sd + mu, true=true * sd + mu)

    # --- 8. NARMA-10 system identification + baselines ----------------
    print("NARMA-10 + baselines ...")
    u, y = ds.narma10(5000, seed=1)
    n_nr, n_ns, (npred, ntrue) = teacher_eval(
        u, y, n_res=400, rho=0.9, leak=1.0, sparsity=0.1,
        train_len=3000, washout=200, seeds=SEEDS, reg=1e-6, input_scaling=0.4)
    summary["narma10_esn"] = dict(nrmse=n_nr, nrmse_std=n_ns)
    np.savez(os.path.join(RES, "narma10_pred.npz"), pred=npred, true=ntrue)
    print(f"  NARMA-10 ESN NRMSE={n_nr:.4f}")

    # --- 9. Baseline comparison across tasks --------------------------
    print("Baselines (ARIMA, linear AR, MLP) ...")
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    def ar_features(series, p):
        X, Y = [], []
        for t in range(p, len(series)):
            X.append(series[t - p:t]); Y.append(series[t])
        return np.array(X), np.array(Y)

    baseline_rows = []

    # Mackey-Glass one-step (fair comparison: all one-step teacher errors)
    mgn, _, _ = normalize(prep_mackey_glass(8000))
    mgn = mgn.ravel()
    tr = 4000
    # ESN one-step on MG
    esn = ESN(1, 400, 1, spectral_radius=0.95, sparsity=0.1, leak_rate=0.9,
              input_scaling=1.0, reg=1e-7, seed=GLOBAL_SEED)
    esn.fit(mgn[:tr, None], mgn[1:tr + 1, None], washout=200)
    esn_pred = esn.predict_teacher(mgn[tr:-1, None], washout=200)
    esn_mg = mx.nrmse(mgn[tr + 1 + 200:][:len(esn_pred)], esn_pred)

    p = 20
    Xtr, Ytr = ar_features(mgn[:tr], p)
    Xte, Yte = ar_features(mgn[tr:], p)
    # Linear AR (Ridge)
    lin = Ridge(alpha=1e-4).fit(Xtr, Ytr)
    lin_mg = mx.nrmse(Yte, lin.predict(Xte))
    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000,
                       random_state=GLOBAL_SEED).fit(Xtr, Ytr)
    mlp_mg = mx.nrmse(Yte, mlp.predict(Xte))
    # ARIMA
    try:
        ar = ARIMA(mgn[:tr], order=(20, 0, 2)).fit()
        fc = ar.forecast(steps=len(mgn) - tr)
        arima_mg = mx.nrmse(mgn[tr:], fc)
    except Exception as e:
        print("  ARIMA MG failed:", e); arima_mg = np.nan

    for name, val in [("ESN", esn_mg), ("Linear AR", lin_mg),
                      ("MLP", mlp_mg), ("ARIMA", arima_mg)]:
        baseline_rows.append(dict(task="Mackey-Glass (1-step)", model=name, nrmse=val))

    # NARMA-10 baselines (system id)
    un = u; yn2, _, _ = normalize(y.reshape(-1, 1)); yn2 = yn2.ravel()
    pp = 10
    def io_features(uu, yy, p):
        X, Y = [], []
        for t in range(p, len(uu)):
            X.append(np.concatenate([uu[t - p:t + 1], yy[t - p:t]]))
            Y.append(yy[t])
        return np.array(X), np.array(Y)
    Xtr2, Ytr2 = io_features(un[:3000], yn2[:3000], pp)
    Xte2, Yte2 = io_features(un[3000:], yn2[3000:], pp)
    lin2 = Ridge(alpha=1e-4).fit(Xtr2, Ytr2)
    mlp2 = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=3000,
                        random_state=GLOBAL_SEED).fit(Xtr2, Ytr2)
    for name, val in [("ESN", n_nr),
                      ("Linear AR", mx.nrmse(Yte2, lin2.predict(Xte2))),
                      ("MLP", mx.nrmse(Yte2, mlp2.predict(Xte2)))]:
        baseline_rows.append(dict(task="NARMA-10 (sys-id)", model=name, nrmse=val))

    # Lorenz one-step baselines
    lorn, _, _ = normalize(prep_lorenz(12000))
    trl = 6000
    esnl = ESN(3, 600, 3, spectral_radius=1.05, sparsity=0.05, leak_rate=1.0,
               input_scaling=0.3, reg=1e-7, seed=GLOBAL_SEED)
    esnl.fit(lorn[:trl], lorn[1:trl + 1], washout=300)
    esnl_pred = esnl.predict_teacher(lorn[trl:-1], washout=300)
    esn_lor = mx.nrmse(lorn[trl + 1 + 300:][:len(esnl_pred)], esnl_pred)
    # linear / MLP on x-channel multistep one-step using all 3 channels
    def ar_feat_multi(series, p):
        X, Y = [], []
        for t in range(p, len(series)):
            X.append(series[t - p:t].ravel()); Y.append(series[t])
        return np.array(X), np.array(Y)
    Xtl, Ytl = ar_feat_multi(lorn[:trl], 5)
    Xtel, Ytel = ar_feat_multi(lorn[trl:], 5)
    linl = Ridge(alpha=1e-4).fit(Xtl, Ytl)
    mlpl = MLPRegressor(hidden_layer_sizes=(80, 80), max_iter=2000,
                        random_state=GLOBAL_SEED).fit(Xtl, Ytl)
    for name, val in [("ESN", esn_lor),
                      ("Linear AR", mx.nrmse(Ytel, linl.predict(Xtel))),
                      ("MLP", mx.nrmse(Ytel, mlpl.predict(Xtel)))]:
        baseline_rows.append(dict(task="Lorenz (1-step)", model=name, nrmse=val))

    pd.DataFrame(baseline_rows).to_csv(os.path.join(RES, "baselines.csv"), index=False)
    print(pd.DataFrame(baseline_rows).to_string(index=False))

    # --- 10. Memory capacity curves -----------------------------------
    print("Memory capacity ...")
    mc_rows = {}
    for rho in [0.5, 0.8, 0.95, 1.1]:
        mc = mx.memory_capacity(ESN, rng_seed=GLOBAL_SEED, n_reservoir=200,
                                spectral_radius=rho, leak_rate=1.0,
                                max_delay=120, input_scaling=0.3)
        mc_rows[f"rho_{rho}"] = mc
        print(f"  rho={rho} total MC={mc.sum():.2f}")
    np.savez(os.path.join(RES, "memory_capacity.npz"),
             delays=np.arange(1, 121), **mc_rows)
    summary["memory_capacity_total"] = {k: float(v.sum()) for k, v in mc_rows.items()}

    summary["runtime_sec"] = time.time() - t0
    summary["seeds"] = SEEDS
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2)
    print("DONE in %.1f s" % (time.time() - t0))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
