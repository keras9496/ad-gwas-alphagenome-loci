"""Figures 5–6 (1e-3 cell-context divergence, 1e-5 direction profiles) and a visual abstract. Values come from result files or are recomputed from the same inputs.
Outputs: paper/figures/fig5.*, fig6.*, visual_abstract.*, paper/numbers2.json
"""
import warnings; warnings.filterwarnings("ignore")
import re, json, sys, io, contextlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from scipy import stats
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RES = EXP/"results"; FIG = HERE/"figures"; W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9994", "#e4e3df"; C_EXP, C_SPL = "#2a78d6", "#eb6834"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "legend.frameon": False, "savefig.dpi": 300})
SUPER = {"뇌 조직": "Brain", "신경세포": "Brain", "교세포": "Brain", "골수계": "Immune", "림프계": "Immune", "혈액·면역 혼합": "Immune", "간": "Liver/metabolic", "대사·내분비": "Liver/metabolic",
         "혈관": "Vascular/muscle", "근육·심장": "Vascular/muscle", "섬유아·중간엽": "Fibroblast", "상피·기타 장기": "Epithelial/other"}
SORD = ["Brain", "Immune", "Liver/metabolic", "Vascular/muscle", "Fibroblast", "Epithelial/other"]; SCOL = dict(zip(SORD, PAL))
def save(fig, name): fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG/f"{name}.png", bbox_inches="tight"); plt.close(fig)
N = {}
# ======================= Fig 5: cell-context divergence (1e-3) =======================
src = (EXP/"scripts"/"e08_entropy.py").read_text(encoding="utf-8"); g = {"__file__": str(EXP/"scripts"/"e08_entropy.py")}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src[:src.index('P("\\n=== G1')], g)
    R1, mo1, mn1, p1 = g["G"](); R2, mo2, mn2, p2 = g["G"]("lab_cm", 2 * g["NC"])
D, CL, PIP = g["D"], g["CL"], g["PIP"]; N["g1"] = dict(obs=mo1, null=float(mn1.mean()), p=p1, n=len(R1)); N["g2"] = dict(obs=mo2, null=float(mn2.mean()), p=p2)
fig = plt.figure(figsize=(7.2, 5.0)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1, 1.1], hspace=.45, wspace=.35)
ax = fig.add_subplot(gs[0, 0]); ax.scatter(R1.H_null, R1.H, s=14, color=C_EXP, alpha=.8, lw=0); lim = [0, max(R1.H.max(), R1.H_null.max()) * 1.05]
ax.plot(lim, lim, color=INK2, lw=.6, ls=(0, (2, 2))); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Null entropy (position-matched)"); ax.set_ylabel("Observed entropy")
ax.text(.03, .97, f"{int((R1.H > R1.H_null).sum())}/{len(R1)} loci above diagonal\n(less concentrated than\nposition-matched sets)", transform=ax.transAxes, va="top", fontsize=7, color=INK2)
ax.set_title("a  Credible-set variants per locus", loc="left", fontsize=8)
ax = fig.add_subplot(gs[0, 1])
for i, (lab, o, n_) in enumerate([("Cell context\n(12 classes)", mo1, mn1), ("Cell × mechanism\n(24 classes)", mo2, mn2)]):
    ax.hist(n_, bins=30, color=GRID, orientation="vertical", bottom=0) if False else None
    ax.errorbar(i, n_.mean(), yerr=[[n_.mean() - np.quantile(n_, .025)], [np.quantile(n_, .975) - n_.mean()]], fmt="o", color=MUTED, capsize=3, ms=5, label="null (95% range)" if i == 0 else None)
    ax.scatter(i + .18, o, color=C_EXP, s=36, zorder=3, label="observed" if i == 0 else None)
ax.set_xticks([0.09, 1.09]); ax.set_xticklabels(["Cell context\n(12 classes)", "Cell × mechanism\n(24 classes)"]); ax.set_xlim(-.5, 1.6); ax.set_ylabel("Mean PIP-weighted entropy")
ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0); ax.set_title(f"b  Mean entropy vs null (P = {p1:.2f}, {p2:.2f})", loc="left", fontsize=8)
# c: composition of top loci by effective variants
ax = fig.add_subplot(gs[1, :]); sel = R1.sort_values("neff", ascending=False).head(22).sort_values("H")
comp = []
for lc in sel.locus:
    pp = PIP[PIP.locus == lc].merge(D[["atlas_variant", "lab"]], on="atlas_variant"); w = pp.pip / pp.pip.sum()
    s_ = pd.Series(w.to_numpy(), index=pp.lab.map(lambda k: SUPER[CL[k]])).groupby(level=0).sum().reindex(SORD).fillna(0); comp.append(s_)
comp = pd.DataFrame(comp, index=sel.locus.str.split("_").str[0]); left = np.zeros(len(comp))
for c in SORD:
    ax.barh(np.arange(len(comp)), comp[c], left=left, color=SCOL[c], height=.78, edgecolor="white", lw=.8, label=c); left += comp[c].to_numpy()
ax.set_yticks(np.arange(len(comp))); ax.set_yticklabels([f"{l} (H {h:.2f})" for l, h in zip(comp.index, sel.H)], fontsize=6.3); ax.set_xlim(0, 1); ax.set_xlabel("PIP-weighted share of predicted cellular labels")
ax.legend(ncol=6, fontsize=6.3, loc="upper center", bbox_to_anchor=(.45, -.2), handlelength=1); ax.set_title("c  Predicted cellular labels of credible-set variants (22 loci with most effective variants)", loc="left", fontsize=8)
save(fig, "fig5")
# ======================= Fig 6: direction profiles (1e-5) =======================
T = (RES/"e11_direction.txt").read_text(encoding="utf-8"); Rc = pd.read_csv(RES/"e11_snp_class.csv")
m0 = re.search(r"소수 부호 비율 중앙 ([\d.]+), 평균 ([\d.]+) \| 0 \(전부 같은 부호\) ([\d.]+)% \| ≥ 10 % ([\d.]+)% \| ≥ 25 % ([\d.]+)%", T); N["d0"] = [float(x) for x in m0.groups()]
strata = re.findall(r"^\s+(r² ≥ 0.5[^\n]*?|0.1 ≤ r² < 0.5|r² < 0.1[^\n]*?)\s+좌위 \d+, SNP\s+(\d+) \| 평균 ([+-][\d.]+) vs 귀무 ([+-][\d.]+) \(95 % ([+-][\d.]+)\) \| 단측 p ([\d.]+)", T, re.M)
d1 = re.search(r"CS 구성원\s+좌위 (\d+), SNP\s+(\d+) \| 평균 ([+-][\d.]+) vs 귀무 ([+-][\d.]+) \(95 % ([+-][\d.]+)\) \| 단측 p ([\d.]+)", T); N["d1"] = [float(x) for x in d1.groups()]
N["d3"] = [[s[0], int(s[1])] + [float(x) for x in s[2:]] for s in strata]
# recompute D0 distribution + D4 + TREM2 example
V = pd.read_parquet(X/"signed"/"variants.parquet"); Kc = ["atlas_variant", "variant_id", "locus", "pos", "chi2", "ref", "alt", "effect_allele", "beta"]
A = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[Kc], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[Kc]]).drop_duplicates("atlas_variant"); V = V.merge(A.drop(columns=["locus"]), on="atlas_variant")
V["s"] = np.where(V.effect_allele == V.alt, np.sign(V.beta), -np.sign(V.beta)); SV = V.set_index("atlas_variant").s
Z = [np.load(f, allow_pickle=True) for f in sorted((X/"signed").glob("part_*.npz"))]; Q = np.vstack([z["Q"] for z in Z]).astype(np.float32); vv = np.concatenate([z["variant"] for z in Z]); gg = np.concatenate([z["gene"] for z in Z])
df = pd.DataFrame({"v": vv, "g": gg, "i": np.arange(len(vv))}).drop_duplicates(["v", "g"]); mx = np.abs(Q).max(1); df["mx"] = mx[df.i]
topi = df.sort_values("mx").groupby("v").tail(1).set_index("v").i
mins = []
for v, i in topi.items():
    q = Q[i]; m = np.abs(q) >= .5
    if m.sum() >= 5: pos = (q[m] > 0).mean(); mins.append(min(pos, 1 - pos))
mins = np.array(mins)
PIPc = PIP.copy(); d4 = []
for lc, gp in PIPc.groupby("locus"):
    lv = V[(V.locus == lc) & V.role.str.contains("lead")].atlas_variant
    for c, gc in gp.groupby("cs"):
        if len(lv) and lv.iat[0] in set(gc.atlas_variant): continue
        rep = gc.sort_values("pip").atlas_variant.iat[-1]; hit = Rc[(Rc.locus == lc) & (Rc.variant == rep)]
        if len(hit): d4.append(dict(locus=lc, rsid=hit.rsid.iat[0], rho=hit.rho.iat[0], pct=hit.pct.iat[0], r2=hit.r2.iat[0]))
d4 = pd.DataFrame(d4); N["d4"] = dict(n=len(d4), n95=int((d4.pct >= .95).sum()), binom_p=float(stats.binomtest(int((d4.pct >= .95).sum()), len(d4), .05, alternative="greater").pvalue))
GENE = {"L008": "TREM2 region", "L014": "SLC24A4", "L002": "PICALM", "L006": "MS4A", "L023": "PLCG2", "L028": "KAT8"}
fig = plt.figure(figsize=(7.4, 6.0)); gs = fig.add_gridspec(2, 2, hspace=.75, wspace=.48)
ax = fig.add_subplot(gs[0, 0]); ax.hist(mins, bins=np.linspace(0, .5, 26), color=C_EXP, edgecolor="white", lw=.6)
ax.set_xlabel("Minority-sign fraction across cell tracks\n(own top gene, |quantile| ≥ 0.5)"); ax.set_ylabel("SNVs")
ax.set_title(f"a  Signed predictions vary across tracks\n    ({N['d0'][3]:.0f}% of SNVs ≥ 10% minority sign)", loc="left", fontsize=8)
ax = fig.add_subplot(gs[0, 1]); labs = ["CS members\n(all)"] + ["high LD\nr² ≥ 0.5", "moderate\n0.1–0.5", "low LD\nr² < 0.1"]
vals = [N["d1"]] + [[0, 0] + s[2:] for s in N["d3"]]
for i, v in enumerate(vals):
    obs, nul, q95, p = (v[2], v[3], v[4], v[5])
    ax.plot([i, i], [nul, q95], color=GRID, lw=6, solid_capstyle="butt"); ax.scatter(i, obs, color=C_EXP if i else INK, s=30, zorder=3); ax.text(i, max(obs, q95) + .003, f"P={p:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.axhline(0, color=INK2, lw=.6); ax.set_ylim(-.012, .062); ax.set_xlim(-.7, 3.5); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=6.8); ax.set_ylabel("Mean direction-profile ρ with lead")
ax.set_title("b  Profile correlation with the lead\n    (grey bar: null mean to 95th pct.)", loc="left", fontsize=8)
ax = fig.add_subplot(gs[1, 0]); d4s = d4.sort_values("pct").reset_index(drop=True)
ax.scatter(np.arange(len(d4s)), d4s.pct, s=np.where(d4s.pct >= .95, 30, 12), color=np.where(d4s.pct >= .95, C_SPL, MUTED), zorder=3)
ax.axhline(.95, color=INK2, lw=.6, ls=(0, (2, 2))); ax.set_ylim(-.03, 1.1); ax.set_xticks([]); ax.set_xlabel(f"Secondary credible-set representatives (n = {len(d4s)})"); ax.set_ylabel("Percentile vs position-matched null")
ax.text(.42, .45, "≥ 95th pct.:\n" + "\n".join(f"{GENE.get(r.locus[:4], r.locus[:4])} ({r.rsid}, r² {r.r2:.2f})" for _, r in d4s[d4s.pct >= .95].iloc[::-1].iterrows()), transform=ax.transAxes, va="top", fontsize=6.2, color=INK2)
ax.set_title(f"c  Post hoc screen of secondary signals\n    ({N['d4']['n95']}/{len(d4s)} ≥ 95th pct.; hypotheses)", loc="left", fontsize=8)
ax = fig.add_subplot(gs[1, 1])
lv = V[(V.locus == "L008_chr6_41161k") & V.role.str.contains("lead")].atlas_variant.iat[0]; iv = Rc[(Rc.locus == "L008_chr6_41161k") & (Rc.rsid == "rs4714447")]
tg = iv.gene.iat[0]; ivv = iv.variant.iat[0]; idx = df.set_index(["v", "g"]).i
a = Q[idx[(lv, tg)]] * SV[lv]; b = Q[idx[(ivv, tg)]] * SV[ivv]
TR = pd.read_csv(X/"ct"/"tracks_RNA_SEQ.csv"); src7 = (EXP/"scripts"/"e07_celltype_tests.py").read_text(encoding="utf-8"); g7 = {"re": re}; exec(src7[src7.index("CLASS_RULES = ["):src7.index("TR = pd.read_csv")], g7)
sc_ = TR.biosample_name.astype(str).map(g7["cls"]).map(SUPER)
for c in SORD:
    m = (sc_ == c).to_numpy(); ax.scatter(a[m], b[m], s=7, color=SCOL[c], alpha=.8, lw=0, label=c)
ax.axhline(0, color=GRID, lw=.6); ax.axvline(0, color=GRID, lw=.6); lead_rs = V[V.atlas_variant == lv].variant_id.iat[0]
ax.set_xlabel(f"Lead {lead_rs}: risk-allele effect on {tg}\n(signed quantile; one point = one cell/tissue RNA track)"); ax.set_ylabel("rs4714447 (risk allele)"); ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, handletextpad=.2, markerscale=1.5)
ax.set_title(f"d  TREM2 region: secondary signal (post hoc)\n    (ρ = {iv.rho.iat[0]:.2f}; rs4714447, r² = {iv.r2.iat[0]:.2f} with lead)", loc="left", fontsize=8)
N["trem2"] = dict(lead=lead_rs, gene=tg, rho=float(iv.rho.iat[0]), r2=float(iv.r2.iat[0]))
save(fig, "fig6")
# ======================= Visual abstract =======================
nums = json.loads((HERE/"numbers.json").read_text(encoding="utf-8"))
fig = plt.figure(figsize=(10.0, 6.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")
def box(x, y, w, h, fc): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=fc, lw=1))
# v4 (정재승 교수님 검토 반영): 핵심 문장, 상자 제목·결론 문구 교체, "각자 다른 세포"·"LD 태그 무관" 표현 삭제
ax.text(50, 60.0, "Within Alzheimer's disease risk loci, stronger association is modestly enriched for predicted expression and splicing effects,\nwhereas the available predictions do not resolve a single cellular context for fine-mapping candidates",
        ha="center", va="center", fontsize=10.2, weight="bold", color=INK, linespacing=1.35)
def blocks(x, ytop, ybot, bl, fs, lg=1.8, bg=1.3, bullet=False, head=INK, rest=INK2, italic=(), ha="left"):
    """문단(블록) 안은 좁게(lg), 블록 사이는 넓게(lg + bg). ytop–ybot 구간 가운데에 세로 정렬."""
    tot = sum(len(b) - 1 for b in bl) * lg + (len(bl) - 1) * (lg + bg); y = ytop - max(0, (ytop - ybot) - tot) / 2
    for k, b in enumerate(bl):
        for j, t in enumerate(b):
            kw = dict(fontsize=fs, va="center", ha=ha, color=head if j == 0 else rest, style="italic" if k in italic else "normal")
            if bullet and j == 0: ax.text(x, y, "•", **kw)
            ax.text(x + (1.3 if bullet else 0), y, t, **kw); y -= lg
        y -= bg
box(1.5, 7, 19.5, 46, "#f3f2ef"); ax.text(11.25, 50.3, "Input", ha="center", fontsize=10, weight="bold")
blocks(11.25, 46.5, 10, [["75 AD GWAS regions", "(Bellenguez 2022)"], ["111,446 common SNVs"], ["Uniform-prior fine-mapping", "(no functional priors)"],
                     ["AlphaGenome Atlas", "variant-effect features:", "expression, splicing,", "chromatin, protein,", "conservation"], ["371 cell/tissue", "RNA-seq tracks"]], 7.8, lg=2.0, bg=2.4, ha="center")
ax.add_patch(FancyArrowPatch((21.8, 30), (24.6, 30), arrowstyle="-|>", mutation_scale=14, color=INK2))
box(25.5, 31.5, 44, 21.5, "#eaf2fc")
ax.text(47.5, 50.9, "WHAT IS ENRICHED", ha="center", fontsize=10, weight="bold", color=C_EXP)
ax.text(47.5, 48.5, "predicted expression and splicing effects among more strongly associated variants", ha="center", fontsize=7.6, color=C_EXP)
axi = fig.add_axes([.27, .545, .14, .16])
T2 = (RES/"e05_tests_1e2.txt").read_text(encoding="utf-8"); blk = T2[T2.index("dec"):].split("\n")[1:11]; tab = np.array([[float(x) for x in l.split()[1:]] for l in blk])
axi.plot(range(1, 11), tab[:, 0] - tab[:5, 0].mean(), color=C_EXP, lw=1.8); axi.plot(range(1, 11), tab[:, 2] - tab[:5, 2].mean(), color=C_SPL, lw=1.8); axi.plot(range(1, 11), tab[:, 1] - tab[:5, 1].mean(), color=MUTED, lw=1)
axi.set_xticks([1, 10]); axi.set_yticks([]); axi.set_xlabel("association decile", fontsize=6.5, labelpad=1); axi.tick_params(labelsize=6)
axi.text(10.4, tab[-1, 2] - tab[:5, 2].mean(), "splicing", fontsize=6, color=C_SPL, va="center"); axi.text(10.4, tab[-1, 0] - tab[:5, 0].mean(), "expression", fontsize=6, color=C_EXP, va="center")
blocks(47, 45.8, 33.0, [["Weak within-locus correlation", f"(ρ̄ {nums['e5a']['발현'][0]:+.3f} expression, {nums['e5a']['스플라이싱'][0]:+.3f} splicing)"], ["Highest-PIP > correlated variants (expr.)"],
                        ["Expression persists after genic-position", "adjustment; splicing attenuated"], [f"GTEx status: expr. → eQTL OR {nums['e7']['Ye|S_발현'][0]:.2f},", f"splice → sQTL OR {nums['e7']['Ys|S_스플라이싱'][0]:.2f} (direction not tested)"]],
       6.9, lg=1.65, bg=1.1, bullet=True, rest=INK2)
box(25.5, 7, 44, 21.8, "#fdf0ea")
ax.text(47.5, 26.6, "WHERE REMAINS UNRESOLVED", ha="center", fontsize=10, weight="bold", color=C_SPL)
ax.text(47.5, 24.2, "candidate variants receive different predicted cellular labels", ha="center", fontsize=7.6, color=C_SPL)
axj = fig.add_axes([.27, .15, .13, .165]); cc = comp.iloc[:8]; lf = np.zeros(len(cc))
for c in SORD: axj.barh(np.arange(len(cc)), cc[c], left=lf, color=SCOL[c], height=.8, edgecolor="white", lw=.5); lf += cc[c].to_numpy()
axj.set_xticks([]); axj.set_yticks([]); axj.set_xlabel("predicted cellular labels per locus", fontsize=6.5, labelpad=1); [axj.spines[x].set_visible(False) for x in ["left", "bottom"]]
blocks(47, 21.0, 9.5, [["Predicted-label entropy not lower", f"than position-matched sets ({N['g1']['obs']:.2f} vs {N['g1']['null']:.2f};", f"0/{N['g1']['n']} loci more concentrated)"],
                       ["Direction profiles: no detectable", f"concordance with the lead (ρ̄ {N['d1'][2]:+.3f})"]], 6.9, lg=1.65, bg=1.6, bullet=True, rest=INK2)
ax.add_patch(FancyArrowPatch((70.4, 30), (73.2, 30), arrowstyle="-|>", mutation_scale=14, color=INK2))
box(74, 7, 24.5, 46, "#f3f2ef"); ax.text(86.25, 50.3, "Interpretation", ha="center", fontsize=10, weight="bold")
blocks(86.25, 46.5, 10, [["Sequence-based predictions", "can help prioritize candidate", "molecular effects, but a lead", "variant's cellular annotation", "should not be assigned to an", "entire locus without accounting", "for fine-mapping uncertainty."],
                       ["Post hoc screen:", "6 secondary signals (e.g.,", "TREM2 region) are", "hypotheses for validation."]], 7.6, lg=2.0, bg=3.6, rest=INK, italic=(1,), ha="center")
# 사후 블록은 보조 정보라 한 단계 옅게
for t_ in ax.texts[-4:]: t_.set_color(INK2)
ax.text(50, 2.6, "AlphaGenome Atlas predictions; fine-mapping without functional priors; GTEx v8 eQTL/sQTL status (no test of effect direction or colocalization).", ha="center", fontsize=7, color=MUTED)
save(fig, "visual_abstract")
(HERE/"numbers2.json").write_text(json.dumps(N, ensure_ascii=False, indent=1, default=float), encoding="utf-8"); print(json.dumps(N, ensure_ascii=False, default=float)[:1500])
