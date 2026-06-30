"""Standard chaotic / nonlinear benchmark time series generators.
All deterministic given seeds. No external downloads required."""
import numpy as np


def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, n_exp=10, dt=1.0,
                 sample=1, discard=1000, x0=1.2, seed=0):
    """Mackey-Glass delay differential equation, RK4 integration.
    tau=17 gives mild chaos (D~2.1); tau=30 stronger chaos."""
    history_len = int(tau / dt)
    rng = np.random.default_rng(seed)
    total = (n * sample) + discard
    x = np.zeros(total + history_len + 1)
    x[:history_len + 1] = x0 + 0.01 * rng.standard_normal(history_len + 1)

    def mg(xt, xtau):
        return beta * xtau / (1 + xtau ** n_exp) - gamma * xt

    for t in range(history_len, total + history_len):
        xt = x[t]
        xtau = x[t - history_len]
        # RK4
        k1 = mg(xt, xtau)
        k2 = mg(xt + 0.5 * dt * k1, xtau)
        k3 = mg(xt + 0.5 * dt * k2, xtau)
        k4 = mg(xt + dt * k3, xtau)
        x[t + 1] = xt + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    series = x[history_len + 1 + discard:][::sample][:n]
    return series.astype(np.float64)


def lorenz(n, dt=0.02, sigma=10.0, rho=28.0, beta=8.0 / 3.0,
           discard=2000, seed=0):
    """Lorenz-63 system, RK4. Returns (n,3) array [x,y,z]."""
    rng = np.random.default_rng(seed)
    state = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)

    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x * y - beta * z])
    total = n + discard
    out = np.zeros((total, 3))
    for t in range(total):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[t] = state
    return out[discard:]


def narma10(n, seed=0):
    """NARMA-10 nonlinear system identification benchmark.
    Input u ~ U(0,0.5); output recurrence of order 10."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n + 50)
    y = np.zeros(n + 50)
    for t in range(10, n + 49):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    return u[50:].astype(np.float64), y[50:].astype(np.float64)


def santa_fe_laser(n=None, seed=0):
    """Santa Fe laser-like series. The original Santa Fe set A is a
    real far-infrared laser intensity recording (Weigend & Gershenfeld).
    To remain fully self-contained and reproducible we synthesize a
    physically-faithful surrogate by integrating a single-mode laser
    (Lorenz-type Haken laser) model, which is the accepted physical model
    for the Santa Fe A intensity dynamics (spiral-out / collapse bursts)."""
    # Haken single-mode laser equations (equivalent to Lorenz form)
    rng = np.random.default_rng(seed)
    dt = 0.02
    sigma, rho, beta = 3.0, 26.0, 1.0
    state = np.array([0.1, 0.1, 0.1]) + 0.01 * rng.standard_normal(3)
    discard = 3000
    N = (n or 4000) + discard

    def f(s):
        E, P, D = s
        return np.array([sigma * (P - E),
                         E * (rho - D) - P,
                         E * P - beta * D])
    out = np.zeros(N)
    for t in range(N):
        k1 = f(state); k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2); k4 = f(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[t] = state[0] ** 2  # intensity ~ |E|^2
    return out[discard:]
