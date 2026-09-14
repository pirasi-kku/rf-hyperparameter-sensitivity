# -*- coding: utf-8 -*-
"""
Reproduces the two statistics added to the manuscript in revision round 4:

  1. 95% percentile bootstrap confidence intervals for the cross-dataset mean
     eta-squared of each hyperparameter (Tables 4 and 5, "95% CI" row).
  2. Shaffer's static family-wise correction applied to the six Wilcoxon
     pairwise level comparisons within each hyperparameter (Table 9,
     "Shaffer" column).

Run from the project root:  python3 bootstrap_and_shaffer.py
Reads results_rf_sensitivity/<dataset>/{stats,wilcoxon}_<dataset>.csv.
Standard library only - no pandas, numpy or scipy needed.

The script first re-derives the mean eta-squared values and the uncorrected
Wilcoxon counts already printed in the manuscript; both reproduce exactly,
which is what validates the corrected figures below them.
"""
import csv, os, math, random
from collections import defaultdict

# Results root: works both in the project working folder (results_rf_sensitivity/)
# and in the public repository layout (results/). Override with argv[1].
import sys
_CANDIDATES = ["results_rf_sensitivity", "results",
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")]
ROOT = sys.argv[1] if len(sys.argv) > 1 else next(
    (c for c in _CANDIDATES if os.path.isdir(c)), _CANDIDATES[0])
CLS = ["iris","wine","ionosphere","vehicle","breast_cancer","credit-g","banknote",
       "digits","segment","phoneme","covertype","pol"]
REG = ["diabetes","abalone","kin8nm","elevators","bike_sharing","houses",
       "california_housing","house_sales"]
ALL = CLS + REG
PARAMS = ["max_depth","max_features","min_samples_leaf","min_samples_split","n_estimators"]

# ---------- 1. eta-squared ----------
eta = defaultdict(dict)   # eta[param][dataset] = value
for ds in ALL:
    p = os.path.join(ROOT, ds, "stats_%s.csv" % ds)
    for r in csv.DictReader(open(p)):
        eta[r["parameter"]][ds] = float(r["eta_squared"])

print("=== mean eta^2 (compare with the paper) ===")
print("%-20s %10s %10s" % ("parameter","cls","reg"))
for pm in PARAMS:
    c = sum(eta[pm][d] for d in CLS)/len(CLS)
    r = sum(eta[pm][d] for d in REG)/len(REG)
    print("%-20s %10.4f %10.4f" % (pm,c,r))

# ---------- 2. bootstrap CI on the mean ----------
def boot_ci(vals, B=10000, alpha=0.05, seed=42):
    rng = random.Random(seed); n=len(vals); means=[]
    for _ in range(B):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    means.sort()
    lo = means[int(math.floor((alpha/2)*B))]
    hi = means[int(math.ceil((1-alpha/2)*B))-1]
    return lo, hi

print("\n=== 95% percentile bootstrap CI for the mean eta^2 (B=10,000, seed=42) ===")
print("%-20s %-26s %-26s" % ("parameter","classification (n=12)","regression (n=8)"))
ci = {}
for pm in PARAMS:
    vc=[eta[pm][d] for d in CLS]; vr=[eta[pm][d] for d in REG]
    mc=sum(vc)/len(vc); mr=sum(vr)/len(vr)
    lc,hc=boot_ci(vc); lr,hr=boot_ci(vr)
    ci[pm]=(mc,lc,hc,mr,lr,hr)
    print("%-20s %.3f [%.3f, %.3f]      %.3f [%.3f, %.3f]" % (pm,mc,lc,hc,mr,lr,hr))

# ---------- 3. Shaffer static on the Wilcoxon families ----------
# k=4 levels -> m=C(4,2)=6 hypotheses.  Equality is an equivalence relation, so the
# set of simultaneously true nulls is fixed by a partition of the 4 levels into
# equality classes, and the number of true pairwise nulls is sum_i C(n_i,2):
#   4      -> 6      3+1    -> 3      2+2 -> 2      2+1+1 -> 1      1+1+1+1 -> 0
# hence S(4) = {0,1,2,3,6}.  The step-j threshold is alpha/t_j with
#   t_j = max{ s in S(4) : s <= m-(j-1) }  ->  t = [6,3,3,3,2,1].
T = [6,3,3,3,2,1]
def shaffer(ps, alpha=0.05):
    """ps: list of 6 p-values (NaN -> 1.0). Returns list of booleans, same order."""
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False]*len(ps)
    for rank, i in enumerate(idx):
        if ps[i] <= alpha / T[rank]:
            rej[i] = True
        else:
            break          # step-down: stop at the first non-rejection
    return rej

raw = defaultdict(lambda: defaultdict(int))   # raw[param][pair] = count
adj = defaultdict(lambda: defaultdict(int))
pairs_seen = defaultdict(list)
for ds in ALL:
    rows = list(csv.DictReader(open(os.path.join(ROOT, ds, "wilcoxon_%s.csv" % ds))))
    byparam = defaultdict(list)
    for r in rows:
        byparam[r["parameter"]].append(r)
    for pm, rs in byparam.items():
        assert len(rs)==6, (ds,pm,len(rs))
        ps=[]
        for r in rs:
            v=r["p_value"].strip()
            ps.append(1.0 if v=="" or v.lower()=="nan" else float(v))
        rej = shaffer(ps)
        for r, pv, k in zip(rs, ps, rej):
            key = (r["level_1"], r["level_2"])
            if key not in pairs_seen[pm]: pairs_seen[pm].append(key)
            if r["significant"].strip().lower()=="true": raw[pm][key]+=1
            if k: adj[pm][key]+=1

print("\n=== Wilcoxon counts out of 20: uncorrected vs Shaffer static (alpha=0.05) ===")
print("%-20s %-18s %6s %8s %6s" % ("parameter","level pair","raw","Shaffer","delta"))
for pm in PARAMS:
    for key in sorted(pairs_seen[pm], key=lambda k:-raw[pm][k]):
        r_,a_=raw[pm][key],adj[pm][key]
        print("%-20s %-18s %6d %8d %+6d" % (pm, "%s vs %s"%key, r_, a_, a_-r_))
