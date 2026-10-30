"""Family generator validation."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from generators_multi import (generate, generate_renewal, generate_hawkes,
                            generate_lrd_mginf, weibull_cv)

RATE = 0.02
N = 4000
PASS, FAIL = [], []

def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))

def isi(ts): return np.diff(ts)

def cv_of(x): return float(x.std() / x.mean()) if len(x) > 1 else float("nan")

def fano_of(ts, mult=5.0):
    if len(ts) < 20: return float("nan")
    W = mult * float(np.mean(isi(ts)))
    if not np.isfinite(W) or W <= 0: return float("nan")
    edges = np.arange(0.0, ts[-1] + W, W)
    if len(edges) < 3: return float("nan")
    counts, _ = np.histogram(ts, bins=edges)
    m = counts.mean()
    return float(counts.var() / m) if m > 0 else float("nan")

def acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else 0.0

def dfa_h(ts):
    x = np.asarray(isi(ts), float)
    if len(x) < 200: return float("nan")
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.round(np.logspace(1, np.log10(len(x) / 4.0), 13)).astype(int))
    F, used = [], []
    for s in scales:
        if s < 4: continue
        nseg = len(y) // s
        if nseg < 4: continue
        seg = y[: nseg * s].reshape(nseg, s)
        t = np.arange(s, dtype=float)
        tm = t - t.mean()
        denom = float(np.dot(tm, tm))
        slopes = seg.dot(tm) / denom
        detr = seg - (slopes[:, None] * tm[None, :] + seg.mean(axis=1)[:, None])
        F.append(float(np.sqrt((detr ** 2).mean()))); used.append(s)
    if len(F) < 3: return float("nan")
    return float(np.polyfit(np.log(used), np.log(F), 1)[0])

FAMS = {"poisson": {}, "renewal": {"shape": 0.7},
        "hawkes": {"branching": 0.6, "beta_rel": 0.5}, "lrd": {"hurst": 0.75}}

def section1():
    print("\n=== S1  rate alignment (target %.3f) ===" % RATE)
    rows = {}
    for fam, kw in FAMS.items():
        ts = generate(fam, RATE, N, seed=1, **kw)
        r = (len(ts) - 1) / ts[-1] if len(ts) > 1 else 0.0
        rel = abs(r - RATE) / RATE
        rows[fam] = {"n": int(len(ts)), "rate": float(r), "rel_err": float(rel)}
        print("    %-8s n=%5d  rate=%.5f  rel_err=%5.1f%%" % (fam, len(ts), r, rel*100))
        check("%s rate within 25%%" % fam, rel < 0.25, "%.1f%%" % (rel*100))
    return rows

def section2():
    print("\n=== S2  Renewal CV vs closed form ===")
    out = {}
    for k in (0.5, 1.0, 2.0):
        ts = generate_renewal(RATE, 8000, shape=k, seed=2)
        c = cv_of(isi(ts)); th = weibull_cv(k)
        out[str(k)] = {"cv_measured": c, "cv_theory": th}
        print("    k=%-4s CV_meas=%.4f  CV_theory=%.4f" % (k, c, th))
        check("renewal k=%s CV matches" % k, abs(c - th) < 0.12 * max(th, 0.2), "|diff|=%.4f" % abs(c-th))
    check("k=1 degenerates to Poisson (CV~1)", abs(out["1.0"]["cv_measured"] - 1.0) < 0.08)
    return out

def section3():
    print("\n=== S3  Hawkes clustering ===")
    out = {}
    for n in (0.3, 0.6):
        ts = generate_hawkes(RATE, 6000, branching=n, beta_rel=0.5, seed=3)
        f = fano_of(ts); a = acf1(isi(ts))
        out[str(n)] = {"fano": f, "acf1": a, "branching": n, "n": int(len(ts))}
        print("    n=%s  n_events=%5d  Fano=%.3f  ACF1=%+.4f" % (n, len(ts), f, a))
        check("hawkes n=%s Fano > 1" % n, f > 1.0, "Fano=%.3f" % f)
        check("hawkes n=%s ACF1 > 0" % n, a > 0.0, "ACF1=%+.4f" % a)
    check("larger branching -> larger Fano", out["0.6"]["fano"] > out["0.3"]["fano"],
          "%.2f -> %.2f" % (out["0.3"]["fano"], out["0.6"]["fano"]))
    return out

def section4():
    print("\n=== S4  LRD (M/G/inf) Hurst via DFA ===")
    out = {}
    for H in (0.6, 0.8):
        ts = generate_lrd_mginf(RATE, 6000, hurst=H, seed=4)
        h = dfa_h(ts)
        out[str(H)] = {"target": H, "measured": h, "n": int(len(ts))}
        print("    H_target=%s  H_meas=%.3f  n=%d" % (H, h, len(ts)))
        check("lrd H=%s DFA close" % H, abs(h - H) < 0.20, "|diff|=%.3f" % abs(h-H))
    check("H=0.8 estimate > H=0.6", out["0.8"]["measured"] > out["0.6"]["measured"])
    return out

def section5():
    print("\n=== S5  Cross-family discriminability (key) ===")
    fams = {"poisson": ("poisson", {}),
            "renewal_k05": ("renewal", {"shape": 0.5}),
            "renewal_k20": ("renewal", {"shape": 2.0}),
            "hawkes_n06": ("hawkes", {"branching": 0.6, "beta_rel": 0.5}),
            "lrd_H08": ("lrd", {"hurst": 0.8})}
    stats = {}
    for name, (fam, kw) in fams.items():
        ts = generate(fam, RATE, 6000, seed=5, **kw)
        x = isi(ts)
        stats[name] = {"cv": cv_of(x), "fano": fano_of(ts), "acf1": acf1(x), "H_dfa": dfa_h(ts)}
        s = stats[name]
        print("    %-12s CV=%5.2f  Fano=%7.2f  ACF1=%+.4f  H_dfa=%.3f" % (name, s["cv"], s["fano"], s["acf1"], s["H_dfa"]))
    p = stats["poisson"]
    check("Poisson vs renewal_k05 by CV", abs(p["cv"] - stats["renewal_k05"]["cv"]) > 0.5,
          "%.2f vs %.2f" % (p["cv"], stats["renewal_k05"]["cv"]))
    check("Poisson vs renewal_k20 by CV", abs(p["cv"] - stats["renewal_k20"]["cv"]) > 0.3,
          "%.2f vs %.2f" % (p["cv"], stats["renewal_k20"]["cv"]))
    check("Poisson vs hawkes by Fano", stats["hawkes_n06"]["fano"] > 2 * max(p["fano"], 1e-9),
          "%.2f vs %.2f" % (p["fano"], stats["hawkes_n06"]["fano"]))
    check("Hawkes vs LRD by ACF1",
          abs(stats["hawkes_n06"]["acf1"] - stats["lrd_H08"]["acf1"]) > 0.05,
          "hawkes %+.3f vs lrd %+.3f" % (stats["hawkes_n06"]["acf1"], stats["lrd_H08"]["acf1"]))
    return stats

def main():
    print("=" * 78); print("Step 3  family generator validation"); print("=" * 78)
    r1, r2, r3, r4, r5 = section1(), section2(), section3(), section4(), section5()
    outdir = os.path.join(PROJ, "results"); os.makedirs(outdir, exist_ok=True)
    payload = {"rate": RATE, "n_events": N, "s1_rates": r1, "s2_renewal_cv": r2,
               "s3_hawkes": r3, "s4_lrd": r4, "s5_discriminability": r5,
               "passed": PASS, "failed": FAIL}
    path = os.path.join(outdir, "step3_family_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\nsaved: " + path)
    print("\n" + "=" * 78)
    print("PASSED %d / %d" % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        print("FAILED:")
        for f_ in FAIL: print("  -", f_)
    print("=" * 78)
    return 0 if not FAIL else 1

if __name__ == "__main__":
    sys.exit(main())
