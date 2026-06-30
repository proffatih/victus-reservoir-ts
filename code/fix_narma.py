"""Recompute the NARMA-10 system-identification comparison under the
standard RC protocol: every model predicts y(n) from the EXOGENOUS INPUT
history u only (no access to past outputs y). This is the fair benchmark
on which reservoir computers are conventionally evaluated."""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from esn import ESN
import datasets as ds
import metrics as mx

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SEEDS = list(range(42, 47))

u, y = ds.narma10(5000, seed=1)
u = u.reshape(-1, 1); y = y.reshape(-1, 1)
yn = (y - y.mean()) / y.std()
tr, wash = 3000, 200

# ESN (tuned config), input = u only -> y
errs, store = [], None
for sd in SEEDS:
    e = ESN(1, 800, 1, spectral_radius=0.9, sparsity=0.1, leak_rate=1.0,
            input_scaling=0.2, reg=1e-7, seed=sd)
    e.fit(u[:tr], yn[:tr], washout=wash)
    p = e.predict_teacher(u[tr:], washout=wash)
    errs.append(mx.nrmse(yn[tr + wash:][:len(p)], p))
    if store is None:
        store = (p, yn[tr + wash:][:len(p)])
esn_nrmse = float(np.mean(errs)); esn_std = float(np.std(errs))
np.savez(os.path.join(RES, "narma10_pred.npz"), pred=store[0], true=store[1])

# Baselines: u-history only (fair RC protocol)
def u_features(uu, yy, p):
    X, Y = [], []
    for t in range(p, len(uu)):
        X.append(uu[t - p:t + 1].ravel()); Y.append(yy[t, 0])
    return np.array(X), np.array(Y)

p = 12
Xtr, Ytr = u_features(u[:tr], yn[:tr], p)
Xte, Yte = u_features(u[tr:], yn[tr:], p)
lin = Ridge(alpha=1e-4).fit(Xtr, Ytr)
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=4000,
                   random_state=42).fit(Xtr, Ytr)
lin_nrmse = float(mx.nrmse(Yte, lin.predict(Xte)))
mlp_nrmse = float(mx.nrmse(Yte, mlp.predict(Xte)))

print(f"NARMA-10 (u-only): ESN={esn_nrmse:.4f}  LinAR={lin_nrmse:.4f}  MLP={mlp_nrmse:.4f}")

# update summary.json
summary = json.load(open(os.path.join(RES, "summary.json")))
summary["narma10_esn"] = dict(nrmse=esn_nrmse, nrmse_std=esn_std)
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2)

# update baselines.csv NARMA rows
df = pd.read_csv(os.path.join(RES, "baselines.csv"))
df = df[df.task != "NARMA-10 (sys-id)"]
add = pd.DataFrame([
    dict(task="NARMA-10 (sys-id)", model="ESN", nrmse=esn_nrmse),
    dict(task="NARMA-10 (sys-id)", model="Linear AR", nrmse=lin_nrmse),
    dict(task="NARMA-10 (sys-id)", model="MLP", nrmse=mlp_nrmse),
])
df = pd.concat([df, add], ignore_index=True)
df.to_csv(os.path.join(RES, "baselines.csv"), index=False)

# update table_headline.json
tab = json.load(open(os.path.join(RES, "table_headline.json")))
tab["narma10"] = dict(nrmse=esn_nrmse, nrmse_std=esn_std)
json.dump(tab, open(os.path.join(RES, "table_headline.json"), "w"), indent=2)
print("updated summary, baselines.csv, table_headline.json")
