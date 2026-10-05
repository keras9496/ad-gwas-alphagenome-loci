"""Figures for the 1e/1e-2 manuscript. Values are read from result files (e02/e05 outputs) or recomputed from the same inputs.
Outputs: paper/figures/fig{1..4}.pdf/.png, figS1.pdf/.png, paper/numbers.json (numbers quoted in the text)
"""
import re, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RES = EXP/"results"; FIG = HERE/"figures"; FIG.mkdir(exist_ok=True)
W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a9994", "#e4e3df", "#fcfcfb"
C_EXP, C_SPL = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "legend.frameon": False})
EN = {"발현": "Expression", "조절": "Regulatory", "스플라이싱": "Splicing", "단백질": "Protein", "보존": "Conservation"}
KO = list(EN); COL = {"발현": C_EXP, "스플라이싱": C_SPL, "조절": MUTED, "단백질": MUTED, "보존": MUTED}
def save(fig, name): fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG/f"{name}.png", bbox_inches="tight"); plt.close(fig)
T2, T5 = (RES/"e02_tests.txt").read_text(encoding="utf-8"), (RES/"e05_tests_1e2.txt").read_text(encoding="utf-8")
def grab(text, head, label):
    blk = text[text.index(head):]; m = re.search(rf"{re.escape(label)}\s+ρ̄ ([+-][\d.]+) \((?:귀무 [+-][\d.]+, )?(?:귀무 )?95 % ([+-][\d.]+)\) 단측 p ([\d.]+)", blk); return tuple(float(x) for x in m.groups())
N = {}
# ---------- data ----------
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2", "maf", "ldscore", "tss_dist"]
D = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
MF = pd.concat([pd.read_parquet(f) for f in sorted((X/"mf").glob("part_*.parquet"))]).drop_duplicates("atlas_variant")
D = D.merge(MF, on="atlas_variant").merge(pd.read_parquet(X/"annot"/"vep.parquet"), on="atlas_variant", how="left").sort_values(["locus", "pos"]).reset_index(drop=True)
pct = lambda v: pd.Series(v).rank(pct=True).to_numpy(); F = {}
for f in ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION", "DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"]: F[f] = pct(D[f"mf_MAX_ABS_{f}"].astype(float).abs().fillna(0))
for f in ["MERGED_SPLICING", "ALPHAMISSENSE"]: F[f] = pct(D[f"mf_{f}"].astype(float).abs().fillna(0))
for f in ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]: F[f] = pct(D[f"mf_{f}"].astype(float).fillna(D[f"mf_{f}"].min()))
for f in ["PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"]: F[f] = (D[f"mf_{f}"].astype(float).fillna(0) > 0).astype(float).to_numpy()
CLS = {"발현": ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION"], "조절": ["DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"], "스플라이싱": ["MERGED_SPLICING"],
       "단백질": ["ALPHAMISSENSE", "PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"], "보존": ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]}
for c in KO: D[f"S_{c}"] = np.max(np.column_stack([F[f] for f in CLS[c]]), 1)
N["n_snv"], N["n_loci"] = int(len(D)), int(D.locus.nunique())
# ---------- Fig 1: AVI vs association, per-locus heterogeneity ----------
L = pd.read_csv(RES/"e05_locus_table.csv", index_col=0); L2 = pd.read_csv(RES/"e02_locus.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
z = L2.sort_values("z_avi").z_avi.to_numpy(); nm = L2.sort_values("z_avi").locus.str.split("_").str[0].to_numpy()
cols = [C_EXP if v > 1.96 else ("#c0bfba" if v > -1.96 else "#6b6a66") for v in z]
ax[0].bar(np.arange(len(z)), z, color=cols, width=.8); ax[0].axhline(0, color=INK2, lw=.6)
for y in (1.96, -1.96): ax[0].axhline(y, color=MUTED, lw=.5, ls=(0, (2, 2)))
ax[0].set_xlim(-1, len(z)); ax[0].set_xticks([]); ax[0].set_xlabel("75 AD loci (ordered)"); ax[0].set_ylabel("Locus z: ρ(AVI, χ²) vs null")
ax[0].text(len(z) - 5, z[-1] - .15, ", ".join(nm[-3:][::-1]), ha="right", va="center", fontsize=6, color=INK2)
ax[0].text(4, z[0] + .1, ", ".join(nm[:3]), ha="left", va="center", fontsize=6, color=INK2)
rb, q95, p = grab(T2, "=== E1a", "AVI"); N["e1a"] = dict(rho=rb, null95=q95, p=p); het = re.search(r"이질성 H = ([\d.]+).*?양측 p ([\d.]+)", T2); N["e1b_p"] = float(het.group(2))
N["e1a_pos"] = float(re.search(r"양의 ρ 좌위 (\d+)%", T2[T2.index("=== E1a"):]).group(1))
ax[0].set_title("a  AVI–association relation per locus", loc="left", fontsize=8, color=INK)
y = np.arange(5)[::-1]
for j, c in enumerate(KO):
    a = grab(T2, "=== E1c", f"K_{c}"); b = grab(T5, "=== E5a", c); N.setdefault("e1c", {})[c] = a; N.setdefault("e5a", {})[c] = b
    for off, (r_, q_, p_), mk in [(+.14, a, "o"), (-.14, b, "s")]:
        ax[1].plot([0, q_], [y[j] + off] * 2, color=GRID, lw=3, solid_capstyle="butt", zorder=1)
        ax[1].scatter(r_, y[j] + off, s=22, marker=mk, color=COL[c], zorder=3, edgecolor="white", lw=.6)
ax[1].axvline(0, color=INK2, lw=.6); ax[1].set_yticks(y); ax[1].set_yticklabels([EN[c] for c in KO]); ax[1].set_xlabel("Mean within-locus partial ρ with χ²")
ax[1].scatter([], [], marker="o", color=INK2, s=18, label="attribution (exploratory)"); ax[1].scatter([], [], marker="s", color=INK2, s=18, label="raw feature (confirmatory)")
ax[1].plot([], [], color=GRID, lw=3, label="null 0–95th pct."); ax[1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
ax[1].set_title("b  Which component tracks association", loc="left", fontsize=8, color=INK)
fig.tight_layout(); save(fig, "fig1")
# ---------- Fig 2: dose-response across chi2 deciles ----------
D["dec"] = D.groupby("locus").chi2.rank(pct=True).mul(10).clip(upper=9.999).astype(int)
tab = D.groupby(["locus", "dec"])[[f"S_{c}" for c in KO]].mean(); m = tab.groupby("dec").mean(); se = tab.groupby("dec").sem()
fig, ax = plt.subplots(figsize=(3.4, 2.6)); xs = np.arange(10) + 1
for c in KO:
    v = m[f"S_{c}"] - m[f"S_{c}"].iloc[:5].mean(); s = se[f"S_{c}"]
    if c in ("발현", "스플라이싱"): ax.fill_between(xs, v - s, v + s, color=COL[c], alpha=.15, lw=0)
    ax.plot(xs, v, color=COL[c], lw=2 if c in ("발현", "스플라이싱") else 1, marker="o", ms=3 if c in ("발현", "스플라이싱") else 0)
    ax.text(10.25, v.iloc[-1], EN[c], color=COL[c] if c in ("발현", "스플라이싱") else INK2, va="center", fontsize=7)
ax.axhline(0, color=INK2, lw=.5); ax.set_xticks(xs); ax.set_xlabel("Within-locus χ² decile"); ax.set_ylabel("Mean score − mean of deciles 1–5\n(percentile units)"); ax.set_xlim(.6, 12.2)
ax.grid(axis="y", color=GRID, lw=.5); ax.set_axisbelow(True); fig.tight_layout(); save(fig, "fig2")
N["decile"] = {EN[c]: [float(m[f"S_{c}"].iloc[0]), float(m[f"S_{c}"].iloc[-1])] for c in KO}
# ---------- Fig 3: highest-PIP variant vs correlated variants outside credible sets; position control ----------
CS = pd.read_csv(RES/"e05_cs.csv"); rng = np.random.default_rng(1)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
for j, c in enumerate(KO):
    d = CS[f"d_{c}"].to_numpy(); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)]); lo, hi = np.quantile(bs, [.025, .975])
    yy = 4 - j; jit = rng.uniform(-.18, .18, len(d)); ax[0].scatter(d, yy + jit, s=4, color=COL[c], alpha=.35, lw=0)
    ax[0].plot([lo, hi], [yy, yy], color=INK, lw=1.4); ax[0].scatter(d.mean(), yy, s=26, color=COL[c], edgecolor=INK, lw=.8, zorder=4)
    pm = re.search(rf"{c}\s+평균 Δ ([+-][\d.]+) \(CS (\d+), 양의 Δ (\d+)%\) 단측 p ([\d.]+)", T5); N.setdefault("e5b", {})[c] = [float(pm.group(1)), int(pm.group(2)), float(pm.group(3)), float(pm.group(4))]
ax[0].axvline(0, color=INK2, lw=.6); ax[0].set_yticks(range(5)[::-1]); ax[0].set_yticklabels([EN[c] for c in KO]); ax[0].set_xlabel("Δ score: highest-PIP variant − correlated variants\noutside credible sets (r² ≥ 0.5)")
ax[0].set_title(f"a  Highest-PIP vs correlated variants ({len(CS)} credible sets)", loc="left", fontsize=8, color=INK)
y = np.arange(5)[::-1]
for j, c in enumerate(KO):
    b = N["e5a"][c]; g6 = re.search(rf"{c}\s+ρ̄ ([+-][\d.]+) \(1e-2 E5a [+-][\d.]+\) 단측 p ([\d.]+)", T5[T5.index("=== E6"):]); a6 = float(g6.group(1)); N.setdefault("e6", {})[c] = [a6, float(g6.group(2))]
    ax[1].plot([b[0], a6], [y[j], y[j]], color=GRID, lw=2); ax[1].scatter(b[0], y[j], s=20, facecolor="white", edgecolor=COL[c], lw=1.2, zorder=3); ax[1].scatter(a6, y[j], s=22, color=COL[c], zorder=4)
ax[1].axvline(0, color=INK2, lw=.6); ax[1].set_yticks(y); ax[1].set_yticklabels([EN[c] for c in KO]); ax[1].set_xlabel("Mean within-locus partial ρ with χ²")
ax[1].scatter([], [], s=18, facecolor="white", edgecolor=INK2, label="base covariates"); ax[1].scatter([], [], s=18, color=INK2, label="+ VEP consequence, exon distance"); ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
ax[1].set_title("b  Adjusting for genic position", loc="left", fontsize=8, color=INK); fig.tight_layout(); save(fig, "fig3")
# ---------- Fig 4: GTEx eQTL/sQTL specificity ----------
G = pd.read_parquet(X/"annot"/"gtex.parquet").merge(D[["atlas_variant", "S_발현", "S_스플라이싱", "maf"]], on="atlas_variant")
tss = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[["atlas_variant", "tss_dist"]], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[["atlas_variant", "tss_dist"]]]).drop_duplicates("atlas_variant")
G = G.merge(tss, on="atlas_variant"); G["log_tss"] = np.log10(G.tss_dist + 1); G["Ye"], G["Ys"] = (G.n_eqtl > 0).astype(float), (G.n_sqtl > 0).astype(float)
Zs = G[["S_스플라이싱", "S_발현", "maf", "log_tss"]].apply(lambda v: (v - v.mean()) / v.std()); Xd = sm.add_constant(Zs); OR = {}
for yv in ["Ye", "Ys"]:
    f = sm.Logit(G[yv], Xd).fit(disp=0); ci = f.conf_int()
    for s in ["S_발현", "S_스플라이싱"]: OR[(yv, s)] = (np.exp(f.params[s]), np.exp(ci.loc[s, 0]), np.exp(ci.loc[s, 1]), f.pvalues[s])
N["e7"] = {f"{a}|{b}": list(map(float, v)) for (a, b), v in OR.items()}; N["gtex_n"] = int(len(G)); N["gtex_eq"], N["gtex_sq"] = float(G.Ye.mean()), float(G.Ys.mean())
fig, ax = plt.subplots(figsize=(3.4, 2.4)); yl = []
for i, (yv, s, lab) in enumerate([("Ye", "S_발현", "Expression score → eQTL"), ("Ye", "S_스플라이싱", "Splicing score → eQTL"), ("Ys", "S_스플라이싱", "Splicing score → sQTL"), ("Ys", "S_발현", "Expression score → sQTL")]):
    o, lo, hi, p = OR[(yv, s)]; yy = 3 - i; diag = (yv, s) in [("Ye", "S_발현"), ("Ys", "S_스플라이싱")]; c = COL["발현" if s == "S_발현" else "스플라이싱"]
    ax.plot([lo, hi], [yy, yy], color=c if diag else MUTED, lw=1.4); ax.scatter(o, yy, s=28, color=c if diag else "white", edgecolor=c if diag else MUTED, lw=1.2, zorder=3)
    ax.text(hi * 1.06, yy, f"OR {o:.2f}", va="center", fontsize=6.5, color=INK2); yl.append(lab)
ax.axvline(1, color=INK2, lw=.6); ax.set_xscale("log"); ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(yl); ax.set_xlabel("Odds ratio per SD (95% CI)")
ax.set_xticks([.25, .5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); ax.set_xlim(.2, 6); fig.tight_layout(); save(fig, "fig4")
# ---------- Fig S1: measurement validity ROC ----------
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.4))
ms = D.missense == 1; fpr, tpr, _ = roc_curve(ms[D.msc.notna()], D.loc[D.msc.notna(), "S_단백질"]); ax[0].plot(fpr, tpr, color=INK, lw=1.5)
genic = D.msc.notna() & ~D.msc.isin(["intergenic_variant", "upstream_gene_variant", "downstream_gene_variant", "regulatory_region_variant", "TF_binding_site_variant"])
fpr2, tpr2, _ = roc_curve(D.loc[genic, "splice"] == 1, D.loc[genic, "S_스플라이싱"]); ax[1].plot(fpr2, tpr2, color=C_SPL, lw=1.5)
e4 = re.search(r"AUC ([\d.]+).*?\n.*?AUC ([\d.]+)", T5[T5.index("=== E4"):]); N["e4_auc"] = [float(e4.group(1)), float(e4.group(2))]
N["e4_n"] = [int(ms.sum()), int((genic & (D.splice == 1)).sum())]; N["pilra_pct"] = float(re.search(r"백분위 ([\d.]+)", T5).group(1))
for a, t in zip(ax, [f"a  Missense (n = {N['e4_n'][0]}) by protein score\n    AUC = {N['e4_auc'][0]:.2f}", f"b  Splice-region (n = {N['e4_n'][1]}) by splicing score\n    AUC = {N['e4_auc'][1]:.2f}"]):
    a.plot([0, 1], [0, 1], color=MUTED, lw=.6, ls=(0, (2, 2))); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.set_title(t, loc="left", fontsize=7.5, color=INK); a.set_aspect("equal")
fig.tight_layout(); save(fig, "figS1")
# locus table for the manuscript
LT = L[L.profile != "뚜렷하지 않음"].copy(); LT["profile"] = LT.profile.map({"스플라이싱형": "splicing", "발현형": "expression", "발현+스플라이싱": "both"})
LT[["profile", "z_tilt_발현", "z_tilt_스플라이싱", "lead", "pip", "eqtl_genes", "sqtl_genes"]].rename(columns={"z_tilt_발현": "z_expr", "z_tilt_스플라이싱": "z_splice"}).to_csv(HERE/"table_loci.csv")
tlt = re.findall(r"^\s+(\S+)\s+Δ ([+-][\d.]+) \(귀무 ([+-][\d.]+)\) 양측 p ([\d.]+)", T5[T5.index("상위 1 % 기울기"):], re.M); N["e6_tilt"] = {EN[a]: [float(b), float(c), float(d)] for a, b, c, d in tlt}
e3 = re.findall(r"^\s+(\S+)\s+Δ비율 ([+-][\d.]+) \(귀무 ([+-][\d.]+)\), 양측 p ([\d.]+)", T2, re.M); N["e3"] = {EN[a]: [float(b), float(c), float(d)] for a, b, c, d in e3}
e2 = re.search(r"주 종류 일치율\(전체\)\s+관측 ([\d.]+) vs 귀무 ([\d.]+) \| 단측 p ([\d.]+)", T2); N["e2"] = [float(x) for x in e2.groups()]
N["e3_consistency"] = [float(x) for x in re.search(r"평균 코사인\) ([+-][\d.]+) \(귀무 ([+-][\d.]+)\), 단측 p ([\d.]+)", T2).groups()]
N["attr_share"] = {EN[c]: float(v) for c, v in re.findall(r"(\S+) (0\.\d+)", T2.split("평균 종류 비율:")[1].split("\n")[0])}
(HERE/"numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=1), encoding="utf-8"); print(json.dumps(N, ensure_ascii=False)[:3000])
