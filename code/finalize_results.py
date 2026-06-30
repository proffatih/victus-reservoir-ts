"""Compute the headline table numbers using the BEST autonomous config for
Mackey-Glass (from the spectral sweep) and emit a JSON consumed for the
manuscript table. Also refreshes the best MG autonomous trajectory figure."""
import os, json
import numpy as np
import run_experiments as R
import metrics as mx

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

summary = json.load(open(os.path.join(RES, "summary.json")))
lt_mg = summary["lyapunov_time_steps"]["mackey_glass"]
lt_lor = summary["lyapunov_time_steps"]["lorenz"]

# Best MG autonomous config (near-critical rho from sweep)
nr, ns, vt, vs, pred, true, mu, sd = R.autonomous_eval(
    R.prep_mackey_glass(10000), n_res=800, rho=1.2, leak=1.0, sparsity=0.05,
    train_len=5000, washout=300, horizon=2000, seeds=R.SEEDS,
    input_scaling=0.5, noise=1e-4, short_horizon=int(lt_mg))
mg_best = dict(nrmse=float(nr), nrmse_std=float(ns), vpt_steps=float(vt),
               vpt_std=float(vs), vpt_lyap=float(vt / lt_mg))
np.savez(os.path.join(RES, "mackeyglass_pred.npz"),
         pred=pred * sd + mu, true=true * sd + mu)
print("MG best:", mg_best)

table = dict(
    mackey_glass=mg_best,
    lorenz=summary["lorenz"],
    narma10=summary["narma10_esn"],
)
json.dump(table, open(os.path.join(RES, "table_headline.json"), "w"), indent=2)
print("wrote table_headline.json")
