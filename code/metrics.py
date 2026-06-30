"""Error metrics, valid prediction time, Lyapunov exponent, memory capacity."""
import numpy as np


def nrmse(y_true, y_pred):
    """Normalized RMSE (normalized by std of target)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (np.std(y_true) + 1e-12)


def valid_prediction_time(y_true, y_pred, threshold=0.4):
    """Number of steps before normalized error first exceeds threshold.
    Error normalized by sqrt(mean over channels of variance). Pathak-style."""
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    if y_true.shape[0] < y_true.shape[1]:
        y_true = y_true.T; y_pred = y_pred.T
    var = np.mean(np.var(y_true, axis=0))
    err = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1) / (var * y_true.shape[1] + 1e-12))
    bad = np.where(err > threshold)[0]
    return int(bad[0]) if len(bad) else len(err)


def lyapunov_rosenstein(series, emb_dim=5, tau=1, fs=1.0, min_tsep=None,
                        traj_len=20):
    """Largest Lyapunov exponent via Rosenstein's method (uses nolds)."""
    import nolds
    return nolds.lyap_r(np.asarray(series).ravel(), emb_dim=emb_dim, lag=tau,
                        min_tsep=min_tsep, trajectory_len=traj_len)


def lyapunov_mackey_glass(tau=17, beta=0.2, gamma=0.1, n_exp=10, dt=0.1,
                          N=200000, discard=20000, renorm=50, seed=0):
    """Largest Lyapunov exponent of the Mackey-Glass DDE via the
    variational (tangent-space) method with Benettin renormalization.
    Returns lambda_1 per unit time (= per sampling step, since the series
    is sampled at unit time step)."""
    H = int(round(tau / dt))
    rng = np.random.default_rng(seed)
    x = np.zeros(N + H + 1); x[:H + 1] = 1.2 + 0.01 * rng.standard_normal(H + 1)
    d = np.zeros(N + H + 1); d[:H + 1] = 1e-8 * rng.standard_normal(H + 1)

    def f(xt, xtau):
        return beta * xtau / (1 + xtau ** n_exp) - gamma * xt

    def df(xt, xtau, dxt, dxtau):
        num = beta * (1 + (1 - n_exp) * xtau ** n_exp) / (1 + xtau ** n_exp) ** 2
        return -gamma * dxt + num * dxtau

    s = 0.0; cnt = 0; d0 = np.linalg.norm(d[:H + 1])
    for t in range(H, N + H):
        xt, xtau = x[t], x[t - H]
        k1 = f(xt, xtau); k2 = f(xt + 0.5 * dt * k1, xtau)
        k3 = f(xt + 0.5 * dt * k2, xtau); k4 = f(xt + dt * k3, xtau)
        x[t + 1] = xt + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        dxt, dxtau = d[t], d[t - H]
        m1 = df(xt, xtau, dxt, dxtau); m2 = df(xt, xtau, dxt + 0.5 * dt * m1, dxtau)
        m3 = df(xt, xtau, dxt + 0.5 * dt * m2, dxtau); m4 = df(xt, xtau, dxt + dt * m3, dxtau)
        d[t + 1] = dxt + dt / 6 * (m1 + 2 * m2 + 2 * m3 + m4)
        if t > H + discard and t % renorm == 0:
            nrm = np.linalg.norm(d[t - H + 1:t + 2])
            if nrm > 0:
                s += np.log(nrm / d0); cnt += 1
                d[t - H + 1:t + 2] *= d0 / nrm
    return s / (cnt * renorm * dt)


def memory_capacity(esn_cls, rng_seed=0, n_reservoir=200, spectral_radius=0.9,
                    leak_rate=1.0, max_delay=200, T=3000, washout=200,
                    input_scaling=1.0, reg=1e-8):
    """Linear short-term memory capacity (Jaeger 2002).
    Drive reservoir with i.i.d. uniform input; target = input delayed by k.
    MC = sum_k r^2(delay k)."""
    rng = np.random.default_rng(rng_seed)
    u = rng.uniform(-0.8, 0.8, T + max_delay + washout)
    esn = esn_cls(n_inputs=1, n_reservoir=n_reservoir, n_outputs=1,
                  spectral_radius=spectral_radius, sparsity=0.1,
                  leak_rate=leak_rate, input_scaling=input_scaling,
                  reg=reg, seed=rng_seed)
    # harvest states once
    inp = u[:, None]
    states = esn._harvest(inp, washout)
    ext = np.hstack([np.ones((states.shape[0], 1)), inp[washout:], states])
    base = washout
    mc_k = []
    for k in range(1, max_delay + 1):
        # target = u delayed by k, aligned with harvested window
        target = u[base - k: base - k + states.shape[0]]
        # ridge solve
        A = ext.T @ ext + reg * np.eye(ext.shape[1])
        w = np.linalg.solve(A, ext.T @ target)
        pred = ext @ w
        # squared correlation coefficient
        c = np.corrcoef(pred, target)[0, 1]
        mc_k.append(max(0.0, c ** 2) if np.isfinite(c) else 0.0)
    return np.array(mc_k)
