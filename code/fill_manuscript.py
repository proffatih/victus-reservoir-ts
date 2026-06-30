r"""Substitute \RES* and \BL* placeholders in manuscript.tex with real
numbers from results/table_headline.json and results/baselines.csv."""
import os, json, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
MAN = os.path.join(HERE, "..", "manuscript")

tab = json.load(open(os.path.join(RES, "table_headline.json")))
mg, lor, nar = tab["mackey_glass"], tab["lorenz"], tab["narma10"]

repl = {
    r"\RESmgnrmse": f"{mg['nrmse']:.2f}$\\pm${mg['nrmse_std']:.2f}",
    r"\RESmgvpt":   f"{mg['vpt_steps']:.0f}$\\pm${mg['vpt_std']:.0f}",
    r"\RESmgvptl":  f"{mg['vpt_lyap']:.2f}",
    r"\RESlornrmse": f"{lor['nrmse']:.2f}$\\pm${lor['nrmse_std']:.2f}",
    r"\RESlorvpt":   f"{lor['vpt_steps']:.0f}$\\pm${lor['vpt_std']:.0f}",
    r"\RESlorvptl":  f"{lor['vpt_lyap']:.2f}",
    r"\RESnarma":    f"{nar['nrmse']:.3f}$\\pm${nar['nrmse_std']:.3f}",
}
# baseline table tokens
bl = pd.read_csv(os.path.join(RES, "baselines.csv"))
def g(task, model):
    s = bl[(bl.task == task) & (bl.model == model)]
    if not len(s):
        return "---"
    v = float(s.nrmse.values[0])
    if v < 1e-2:
        m, e = f"{v:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{v:.3f}"
repl.update({
    r"\BLmgesn": g("Mackey-Glass (1-step)", "ESN"),
    r"\BLmglin": g("Mackey-Glass (1-step)", "Linear AR"),
    r"\BLmgmlp": g("Mackey-Glass (1-step)", "MLP"),
    r"\BLmgari": g("Mackey-Glass (1-step)", "ARIMA"),
    r"\BLloresn": g("Lorenz (1-step)", "ESN"),
    r"\BLlorlin": g("Lorenz (1-step)", "Linear AR"),
    r"\BLlormlp": g("Lorenz (1-step)", "MLP"),
    r"\BLnaresn": g("NARMA-10 (sys-id)", "ESN"),
    r"\BLnarlin": g("NARMA-10 (sys-id)", "Linear AR"),
    r"\BLnarmlp": g("NARMA-10 (sys-id)", "MLP"),
})

src = open(os.path.join(MAN, "manuscript.tex")).read()
for k, v in repl.items():
    src = src.replace(k, v)
# remove the placeholder note from the caption
src = src.replace(" Placeholder tokens are filled automatically from\n\\texttt{results/}.", "")
open(os.path.join(MAN, "manuscript.tex"), "w").write(src)
print("filled placeholders:", repl)
