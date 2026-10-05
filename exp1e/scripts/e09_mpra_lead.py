"""실험 1e-4 e09: CS 안 Atlas 발현 최고 변이 vs GWAS 선도 SNP — 미세아교세포 MPRA emVar (PROTOCOL_exp1e4_2026-09-27).
H1 정확 McNemar(iMGL 주, HMC3 재현) · H2 CS 조건부 로지스틱 · H3 무작위 구성원 대비 · H4 좌위 전반 AUC · 민감도 · 대식세포/HEK 기술.
출력: results/e09_mpra_lead.txt, e09_cs_pairs.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"; W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e"); M = Path("D:/AD_GWAS_Atlas_data/mpra")
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
# ---------- Atlas 점수 (1e-2 정의) ----------
K = ["atlas_variant", "variant_id", "locus", "chi2"]
D = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
MF = pd.concat([pd.read_parquet(f) for f in sorted((X/"mf").glob("part_*.parquet"))]).drop_duplicates("atlas_variant"); D = D.merge(MF, on="atlas_variant")
pct = lambda v: pd.Series(v).rank(pct=True).to_numpy()
mx = lambda fs, pre="mf_MAX_ABS_": np.max(np.column_stack([pct(D[f"{pre}{f}"].astype(float).abs().fillna(0)) for f in fs]), 1)
D["S_expr"] = mx(["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION"]); D["S_reg"] = mx(["DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"])
D["S_spl"] = pct(D.mf_MERGED_SPLICING.astype(float).abs().fillna(0)); D["S_cons"] = np.max(np.column_stack([pct(D[f"mf_{f}"].astype(float).fillna(D[f"mf_{f}"].min())) for f in ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]]), 1)
D["S_es"] = np.maximum(D.S_expr, D.S_spl)
G = pd.concat([pd.read_parquet(f) for f in sorted((X/"ct").glob("rna_meta_*.parquet"))]).drop_duplicates("atlas_variant").set_index("atlas_variant").gene
D["gene"] = D.atlas_variant.map(G)
lead = D.sort_values("chi2").groupby("locus").tail(1).set_index("locus").variant_id
# ---------- CS ----------
PIP = pd.concat([pd.read_csv(f).assign(locus=f.name.replace("_pip.csv", "")) for d in [W/"s2"/"out_P0", W/"s2_rep25"/"out_P0"] for f in d.glob("*_pip.csv")])
C = PIP[PIP.cs.notna()].merge(D.drop(columns=["locus"]), on="atlas_variant"); C["csid"] = C.locus + "|" + C.cs.astype(int).astype(str)
# ---------- MPRA ----------
def lee(nm, rule="main"):
    d = pd.read_csv(M/"lee2025"/f"{nm}_MPRA_AllelicActivity.tsv.gz", sep="\t").drop_duplicates("rsID").set_index("rsID")
    ev = {"main": (d.CRS == "active") & (d.fdr < .05), "fdr_only": d.fdr < .05, "fdr10": (d.CRS == "active") & (d.fdr < .10)}[rule]
    return ev.astype(int)
B = pd.read_csv(M/"bond_S4_allelic.csv").drop_duplicates("RSID").set_index("RSID").emVar.astype(int)
Cp = pd.read_csv(M/"cooper_S1_mpra.csv").drop_duplicates("rsID").set_index("rsID"); Cp = (Cp.q < .05).astype(int)
def pairs(ev, score="S_expr", among_tested=False):
    rows = []
    for cid, g in C.groupby("csid"):
        lc = g.locus.iat[0]; ld = lead.get(lc)
        if len(g) < 2 or ld not in set(g.variant_id): continue
        pool = g[g.variant_id.isin(ev.index)] if among_tested else g
        if pool.empty: continue
        at = pool.sort_values(score).variant_id.iat[-1]
        if at == ld or at not in ev.index or ld not in ev.index: continue
        r = g.set_index("variant_id")
        rows.append(dict(csid=cid, locus=lc, n=len(g), lead=ld, atlas_top=at, ev_top=int(ev[at]), ev_lead=int(ev[ld]), pip_top=r.at[at, "pip"], pip_lead=r.at[ld, "pip"],
                         S_top=r.at[at, score], S_lead=r.at[ld, score], gene_top=r.at[at, "gene"]))
    return pd.DataFrame(rows)
def mcnemar(R):
    b = int(((R.ev_top == 1) & (R.ev_lead == 0)).sum()); c = int(((R.ev_top == 0) & (R.ev_lead == 1)).sum())
    p = stats.binomtest(b, b + c, .5, alternative="greater").pvalue if b + c else np.nan; return b, c, p
def report(tag, R):
    b, c, p = mcnemar(R); P(f"  {tag:40s} CS {len(R):3d} | emVar: Atlas 최고 {R.ev_top.mean():.2f} vs 선도 {R.ev_lead.mean():.2f} | b(Atlas만) {b}, c(선도만) {c} | 단측 p {p:.4f}"); return p
# ---------- H1 ----------
P("=== H1 (주): CS 안 Atlas 발현 최고 변이 vs GWAS 선도 SNP — 미세아교세포 MPRA emVar (CRS active & fdr < 0.05) ===")
EV = {"iMGL": lee("iMGL"), "HMC3": lee("HMC3")}
R1 = pairs(EV["iMGL"]); p1 = report("iMGL (주)", R1); R1h = pairs(EV["HMC3"]); p1h = report("HMC3 (재현)", R1h)
b1, c1, _ = mcnemar(R1); bh, ch, _ = mcnemar(R1h)
v = "H1 통과" if p1 < .05 else "H1 불통과"; rep = "HMC3 재현" if (p1h < .05 and bh > ch) else ("HMC3 같은 방향(비유의)" if bh > ch else "HMC3 재현 안 됨")
P(f"  → {v} | {rep}")
R1.merge(R1h[["csid", "ev_top", "ev_lead"]].rename(columns={"ev_top": "ev_top_HMC3", "ev_lead": "ev_lead_HMC3"}), on="csid", how="outer").to_csv(RES/"e09_cs_pairs.csv", index=False)
P("  사례 (iMGL 또는 HMC3에서 Atlas 최고만 emVar):")
for r in R1.merge(R1h[["csid", "ev_top", "ev_lead"]], on="csid", how="left", suffixes=("", "_h")).itertuples():
    if (r.ev_top == 1 and r.ev_lead == 0) or (getattr(r, "ev_top_h", 0) == 1 and getattr(r, "ev_lead_h", 0) == 0):
        P(f"    {r.locus:20s} Atlas 최고 {r.atlas_top} (PIP {r.pip_top:.2f}, 발현 {r.S_top:.3f}, 유전자 {r.gene_top}) vs 선도 {r.lead} (PIP {r.pip_lead:.2f}, 발현 {r.S_lead:.3f}) | iMGL {r.ev_top}/{r.ev_lead}, HMC3 {getattr(r, 'ev_top_h', np.nan)}/{getattr(r, 'ev_lead_h', np.nan)}")
# ---------- H2 ----------
P("\n=== H2 (부): CS 안 순위 — 조건부 로지스틱 emVar ~ 발현 점수(CS 안 표준화), 단측 ===")
for nm, ev in EV.items():
    d = C[C.variant_id.isin(ev.index)].copy(); d["y"] = d.variant_id.map(ev); d = d.groupby("csid").filter(lambda g: len(g) >= 2 and g.y.nunique() == 2)
    d["z"] = d.groupby("csid").S_expr.transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    f = ConditionalLogit(d.y.to_numpy(), d[["z"]].to_numpy(), groups=d.csid.to_numpy()).fit(disp=0)
    P(f"  {nm}: CS {d.csid.nunique()}, 변이 {len(d)}, emVar {int(d.y.sum())} | OR/SD {np.exp(f.params[0]):.2f} (95 % {np.exp(f.conf_int()[0][0]):.2f}–{np.exp(f.conf_int()[0][1]):.2f}), 단측 p {stats.norm.sf(f.tvalues[0]):.4f}")
# ---------- H3 ----------
P("\n=== H3 (부): Atlas 최고 변이 vs 같은 CS 무작위 구성원 (검사된 구성원 중 선택) ===")
rng = np.random.default_rng(20260930)
for nm, ev in EV.items():
    tops, pools = [], []
    for cid, g in C[C.variant_id.isin(ev.index)].groupby("csid"):
        if len(g) < 2: continue
        tops.append(ev[g.sort_values("S_expr").variant_id.iat[-1]]); pools.append(ev.loc[g.variant_id].to_numpy())
    o = np.mean(tops); nl = np.array([np.mean([rng.choice(p_) for p_ in pools]) for _ in range(10000)])
    P(f"  {nm}: CS {len(tops)} | Atlas 최고 emVar {o:.3f} vs 무작위 {nl.mean():.3f} | 단측 p {(1 + (nl >= o).sum()) / (len(nl) + 1):.4f}")
# ---------- H4 ----------
P("\n=== H4 (기술): 75 좌위 전 검사 SNV — 발현 점수의 emVar AUC (좌위 층화 순열) ===")
for nm, ev in EV.items():
    d = D[D.variant_id.isin(ev.index)].drop_duplicates("variant_id").copy(); d["y"] = d.variant_id.map(ev)
    auc = lambda s, y: stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum())
    o = auc(d.S_expr.to_numpy(), d.y.to_numpy()); nl = []
    for _ in range(1000): nl.append(auc(d.groupby("locus").S_expr.transform(lambda v: rng.permutation(v.to_numpy())).to_numpy(), d.y.to_numpy()))
    P(f"  {nm}: SNV {len(d)} (좌위 {d.locus.nunique()}), emVar {int(d.y.sum())} | AUC {o:.3f} (귀무 {np.mean(nl):.3f}), 단측 p {(1 + (np.array(nl) >= o).sum()) / 1001:.4f}")
# ---------- 민감도 ----------
P("\n=== 민감도 (H1, iMGL / HMC3) ===")
for tag, kw, rule in [("emVar = fdr < 0.05 (CRS 무관)", {}, "fdr_only"), ("emVar = active & fdr < 0.10", {}, "fdr10"), ("Atlas 최고 = 발현·스플라이싱 최대", {"score": "S_es"}, "main"),
                      ("Atlas 최고 = 검사된 구성원 중", {"among_tested": True}, "main"), ("(비교) 조절 점수 최고", {"score": "S_reg"}, "main"), ("(비교) 보존 점수 최고", {"score": "S_cons"}, "main")]:
    for nm in ["iMGL", "HMC3"]: report(f"{tag} [{nm}]", pairs(lee(nm, rule), **kw))
P("\n=== 기술: 다른 세포 MPRA (검사 CS 적음) ===")
report("THP-1 대식세포 (Bond emVar)", pairs(B)); report("HEK293T (Cooper q < 0.05)", pairs(Cp))
P("\n=== 기술: PIP 최고 ≠ 선도인 CS ===")
for cid, g in C.groupby("csid"):
    ld = lead.get(g.locus.iat[0]); pt = g.sort_values("pip").variant_id.iat[-1]
    if len(g) >= 2 and ld in set(g.variant_id) and pt != ld:
        P(f"  {cid}: 선도 {ld} (PIP {g.set_index('variant_id').pip[ld]:.2f}) vs PIP 최고 {pt} (PIP {g.pip.max():.2f}) | iMGL emVar 선도 {EV['iMGL'].get(ld, 'NA')} / PIP 최고 {EV['iMGL'].get(pt, 'NA')}")
(RES/"e09_mpra_lead.txt").write_text("\n".join(log), encoding="utf-8")
