"""
Leaky-integrator Echo State Network (ESN) implemented from scratch with NumPy.
Reference formulation: Jaeger (2001), Lukosevicius & Jaeger (2009),
Jaeger (2007, leaky integrator), Maass et al. (2002).

All matrices use fixed seeds for full reproducibility.
"""
import numpy as np


class ESN:
    """Leaky-integrator echo state network with ridge (Tikhonov) readout."""

    def __init__(self, n_inputs, n_reservoir, n_outputs,
                 spectral_radius=0.9, sparsity=0.1, leak_rate=1.0,
                 input_scaling=1.0, bias_scaling=0.0,
                 reg=1e-6, noise=0.0, seed=0):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity            # fraction of NONZERO entries
        self.leak_rate = leak_rate          # alpha in [0,1]
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.reg = reg
        self.noise = noise
        self.seed = seed
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.seed)
        self._noise_rng = np.random.default_rng(self.seed + 99991)
        # Reservoir weight matrix W (sparse, scaled to target spectral radius)
        W = rng.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        mask = rng.uniform(0, 1, (self.n_reservoir, self.n_reservoir)) > self.sparsity
        W[mask] = 0.0
        # spectral radius scaling
        eigs = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigs))
        if rho > 0:
            W *= self.spectral_radius / rho
        self.W = W
        # Input weights
        self.W_in = self.input_scaling * rng.uniform(-1, 1,
                                                      (self.n_reservoir, self.n_inputs))
        # Bias
        self.b = self.bias_scaling * rng.uniform(-1, 1, (self.n_reservoir,))
        self.W_out = None

    def _update(self, state, u, add_noise=False):
        pre = self.W_in @ u + self.W @ state + self.b
        new = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre)
        if add_noise and self.noise > 0:
            new = new + self.noise * self._noise_rng.standard_normal(self.n_reservoir)
        return new

    def _harvest(self, inputs, washout, state0=None, add_noise=False):
        T = inputs.shape[0]
        state = np.zeros(self.n_reservoir) if state0 is None else state0.copy()
        states = np.zeros((T, self.n_reservoir))
        for t in range(T):
            state = self._update(state, inputs[t], add_noise=add_noise)
            states[t] = state
        self.last_state = state
        return states[washout:]

    def fit(self, inputs, targets, washout=100):
        """inputs: (T, n_inputs); targets: (T, n_outputs)."""
        states = self._harvest(inputs, washout, add_noise=True)
        # extended states with input + bias term for readout
        ext = np.hstack([np.ones((states.shape[0], 1)),
                         inputs[washout:], states])
        Y = targets[washout:]
        # ridge regression
        n_feat = ext.shape[1]
        A = ext.T @ ext + self.reg * np.eye(n_feat)
        B = ext.T @ Y
        self.W_out = np.linalg.solve(A, B)
        self._n_feat = n_feat
        return self

    def predict_teacher(self, inputs, washout=100, state0=None):
        """Teacher-forced (one-step) prediction over given inputs."""
        states = self._harvest(inputs, washout, state0=state0)
        ext = np.hstack([np.ones((states.shape[0], 1)),
                         inputs[washout:], states])
        return ext @ self.W_out

    def generate(self, n_steps, last_input, state0):
        """Autonomous (closed-loop) generation: feed back own output.
        Assumes n_inputs == n_outputs (next-step prediction task)."""
        # state0 was produced by consuming last_input; the readout row
        # [1, u_t, x_t] predicts u_{t+1}. So we first read out from the
        # given (input, state) pair, then advance the reservoir with the
        # freshly predicted output.
        state = state0.copy()
        u = last_input.copy()
        out = np.zeros((n_steps, self.n_outputs))
        for t in range(n_steps):
            ext = np.hstack([[1.0], u, state])
            y = ext @ self.W_out
            out[t] = y
            state = self._update(state, y)   # feed prediction back
            u = y
        return out
