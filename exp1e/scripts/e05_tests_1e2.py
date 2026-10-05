"""실험 1e-2 e05: E4(측정 타당성) · E5a(사전 방향 ρ̄) · E5b(CS 인과 후보 vs LD 태그) · E6(위치 교란 통제) · E7(GTEx eQTL/sQTL) · 좌위 목록 (PROTOCOL_exp1e2).
출력: results/e05_tests_1e2.txt, e05_locus_table.csv, e05_cs.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"; W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e")
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2", "maf", "ldscore", "tss_dist"]
D = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
MF = pd.concat([pd.read_parquet(f) for f in sorted((X/"mf").glob("part_*.parquet"))]).drop_duplicates("atlas_variant")
D = D.merge(MF, on="atlas_variant").merge(pd.read_parquet(X/"annot"/"vep.parquet"), on="atlas_variant", how="left").merge(pd.read_parquet(X/"annot"/"exon_dist.parquet"), on="atlas_variant", how="left")
D = D.sort_values(["locus", "pos"]).reset_index(drop=True); D["log_tss"] = np.log10(D.tss_dist + 1); D["log_exon"] = np.log10(D.exon_dist.fillna(1e6) + 1)
P(f"SNV {len(D):,} (좌위 {D.locus.nunique()}); VEP 있음 {D.msc.notna().mean():.1%}")
# ---------- 종류 점수 (원 특성 백분위, 종류 안 최대) ----------
pct = lambda v: pd.Series(v).rank(pct=True).to_numpy()
F = {}
for f in ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION", "DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"]: F[f] = pct(D[f"mf_MAX_ABS_{f}"].astype(float).abs().fillna(0))
for f in ["MERGED_SPLICING", "ALPHAMISSENSE"]: F[f] = pct(D[f"mf_{f}"].astype(float).abs().fillna(0))
for f in ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]: F[f] = pct(D[f"mf_{f}"].astype(float).fillna(D[f"mf_{f}"].min()))
for f in ["PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"]: F[f] = (D[f"mf_{f}"].astype(float).fillna(0) > 0).astype(float).to_numpy()
CLS = {"발현": ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION"], "조절": ["DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"], "스플라이싱": ["MERGED_SPLICING"],
       "단백질": ["ALPHAMISSENSE", "PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"], "보존": ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]}
CN = list(CLS)
for c in CN: D[f"S_{c}"] = np.max(np.column_stack([F[f] for f in CLS[c]]), 1)
# ---------- E4 ----------
def auc(pos, neg):
    u = stats.mannwhitneyu(pos, neg, alternative="greater"); return u.statistic / (len(pos) * len(neg)), u.pvalue
P("\n=== E4 측정 타당성 (양성 대조) ===")
ms = D.missense == 1; a1, p1 = auc(D.loc[ms, "S_단백질"], D.loc[~ms & D.msc.notna(), "S_단백질"])
genic = D.msc.notna() & ~D.msc.isin(["intergenic_variant", "upstream_gene_variant", "downstream_gene_variant", "regulatory_region_variant", "TF_binding_site_variant"])
sp = genic & (D.splice == 1); a2, p2 = auc(D.loc[sp, "S_스플라이싱"], D.loc[genic & (D.splice == 0), "S_스플라이싱"])
spc = genic & (D.splice_core == 1); a2c = auc(D.loc[spc, "S_스플라이싱"], D.loc[genic & (D.splice == 0), "S_스플라이싱"])[0] if spc.sum() else np.nan
pr = D[D.variant_id == "rs1859788"]; q3 = pr["S_단백질"].iat[0] if len(pr) else np.nan
P(f"  (i) 미스센스 {int(ms.sum())} vs 나머지: 단백질 점수 AUC {a1:.3f} (p {p1:.1e})")
P(f"  (ii) 스플라이스 부위 {int(sp.sum())} (그중 공여·수용 {int(spc.sum())}) vs 유전자 내 나머지: 스플라이싱 점수 AUC {a2:.3f} (p {p2:.1e}); 공여·수용만 AUC {a2c:.3f}")
P(f"  (iii) PILRA rs1859788 단백질 점수 백분위 {q3:.3f} (원 AlphaMissense {pr['mf_ALPHAMISSENSE'].iat[0] if len(pr) else np.nan})")
ok4 = (a1 >= .75 and p1 < 1e-3 and a2 >= .75 and p2 < 1e-3 and q3 >= .95); P(f"  → E4 {'통과: 측정 타당' if ok4 else '불통과: 측정 타당성 미확보 (E5–E7 경고)'}")
# ---------- 좌위 ρ 도구 ----------
rng = np.random.default_rng(20260927)
def rres(v, Z):
    rv = stats.rankdata(v); Zr = np.column_stack([np.ones(len(v))] + [stats.rankdata(z) for z in Z.T]); return rv - Zr @ np.linalg.lstsq(Zr, rv, rcond=None)[0]
def rho_bar(col, conf, nperm=1000):
    obs, nul = [], []
    for lc, g in D.groupby("locus", sort=False):
        Z = g[conf].to_numpy(float); Z = Z[:, Z.std(0) > 0]; y = rres(g.chi2.to_numpy(), Z); s = g[col].to_numpy(float); n = len(s)
        obs.append(np.corrcoef(rres(s, Z), y)[0, 1]); sh = rng.integers(max(1, n // 20), n - max(1, n // 20), nperm); nul.append([np.corrcoef(rres(np.roll(s, k), Z), y)[0, 1] for k in sh])
    o, N = np.array(obs), np.array(nul); nb = N.mean(0); return o.mean(), nb, (1 + (nb >= o.mean()).sum()) / (len(nb) + 1), np.mean(o > 0)
BASE = ["ldscore", "maf", "log_tss"]
P("\n=== E5a (주): 원 특성 종류 점수 ρ̄ (사전 방향: 발현·스플라이싱 > 0, 각 단측 α 0.025) ===")
e5a = {}
for c in CN:
    rb, nb, p, fp = rho_bar(f"S_{c}", BASE); e5a[c] = (rb, p)
    tag = " → " + ("통과" if p < .025 else "불통과") if c in ("발현", "스플라이싱") else " (기술)"
    P(f"  {c:6s} ρ̄ {rb:+.4f} (귀무 95 % {np.quantile(nb, .95):+.4f}) 단측 p {p:.4f} | 양의 좌위 {fp:.0%}{tag}")
# ---------- E5b: CS 인과 후보 vs LD 태그 ----------
P("\n=== E5b (주): CS 인과 후보(PIP 최대) − 같은 블록 LD 태그(r² ≥ 0.5, CS 밖) (부호 뒤집기 순열 10,000회; 사전 방향 > 0, 단측 α 0.025) ===")
PIP = pd.concat([pd.read_csv(f).assign(locus=f.name.replace("_pip.csv", "")) for d in [W/"s2"/"out_P0", W/"s2_rep25"/"out_P0"] for f in d.glob("*_pip.csv")])
Dx = D.set_index("atlas_variant"); rows = []
for (lc, cs), g in PIP[PIP.cs.notna()].groupby(["locus", "cs"]):
    lead = g.sort_values("pip").atlas_variant.iat[-1]
    if lead not in Dx.index: continue
    z = np.load(W/"ld"/f"{lc}.npz", allow_pickle=True); v = pd.Index(z["variant"].astype(str)); i = v.get_indexer([lead])[0]
    if i < 0: continue
    r2 = z["R"][i].astype(float) ** 2; incs = set(PIP[(PIP.locus == lc) & PIP.cs.notna()].atlas_variant)
    tags = [t for t, rr in zip(v, r2) if rr >= .5 and t != lead and t not in incs and t in Dx.index]
    if not tags: continue
    rows.append(dict(locus=lc, cs=cs, lead=lead, pip=g.pip.max(), n_tags=len(tags), **{f"d_{c}": Dx.at[lead, f"S_{c}"] - Dx.loc[tags, f"S_{c}"].mean() for c in CN},
                     **{f"lead_{c}": Dx.at[lead, f"S_{c}"] for c in CN}))
CS = pd.DataFrame(rows); CS.to_csv(RES/"e05_cs.csv", index=False)
for c in CN:
    d = CS[f"d_{c}"].to_numpy(); o = d.mean(); nl = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(10000)]); p = (1 + (nl >= o).sum()) / (len(nl) + 1)
    tag = " → " + ("통과" if p < .025 else "불통과") if c in ("발현", "스플라이싱") else " (기술)"
    P(f"  {c:6s} 평균 Δ {o:+.4f} (CS {len(d)}, 양의 Δ {np.mean(d > 0):.0%}) 단측 p {p:.4f}{tag}")
P("  (기술) 좌위 안 χ² 10분위별 평균 점수 (좌위 평균):")
D["dec"] = D.groupby("locus").chi2.rank(pct=True).mul(10).clip(upper=9.999).astype(int)
tab = D.groupby(["locus", "dec"])[[f"S_{c}" for c in CN]].mean().groupby("dec").mean()
P("  " + tab.round(3).rename(columns=lambda x: x[2:]).to_string().replace("\n", "\n  "))
# ---------- E6 ----------
P("\n=== E6: 위치 교란 통제 (VEP 결과 유형 + 엑손 경계 거리 추가 보정) ===")
grp = {"coding": ["missense_variant", "synonymous_variant", "stop_gained", "stop_lost", "start_lost", "coding_sequence_variant", "inframe_insertion", "inframe_deletion", "stop_retained_variant", "incomplete_terminal_codon_variant"],
       "utr": ["5_prime_UTR_variant", "3_prime_UTR_variant"], "splice": sorted({"splice_donor_variant", "splice_acceptor_variant", "splice_region_variant", "splice_donor_5th_base_variant", "splice_donor_region_variant", "splice_polypyrimidine_tract_variant"}),
       "intron": ["intron_variant"], "flank": ["upstream_gene_variant", "downstream_gene_variant"], "nc": ["non_coding_transcript_exon_variant", "mature_miRNA_variant"]}
D["cat"] = "other"
for k, vs in grp.items(): D.loc[D.msc.isin(vs), "cat"] = k
D.loc[D.msc == "intergenic_variant", "cat"] = "intergenic"
dm = pd.get_dummies(D.cat, prefix="cat", dtype=float); D = pd.concat([D, dm], axis=1); CONF6 = BASE + ["log_exon"] + [c for c in dm.columns if c != "cat_intergenic"]
P("  결과 유형 분포: " + ", ".join(f"{k} {v:.1%}" for k, v in D.cat.value_counts(normalize=True).items()))
for c in CN:
    rb, nb, p, fp = rho_bar(f"S_{c}", CONF6); tag = " → " + ("위치로 설명되지 않음" if p < .05 else "위치로 설명될 수 있음") if c in ("발현", "스플라이싱") else " (기술)"
    P(f"  {c:6s} ρ̄ {rb:+.4f} (1e-2 E5a {e5a[c][0]:+.4f}) 단측 p {p:.4f}{tag}")
P("  상위 1 % 기울기 (결과 유형 구성으로 배경 재가중; 좌위 평균, 거리 보존 평행이동 1,000회):")
def tilt(nperm=1000):
    ob, nl, lz = [], [], []
    for lc, g in D.groupby("locus", sort=False):
        pos = g.pos.to_numpy(); n = len(g); k = max(5, int(round(.01 * n))); top = np.sort(np.argsort(-g.chi2.to_numpy())[:k]); Sg = g[[f"S_{c}" for c in CN]].to_numpy(); cat = g.cat.to_numpy()
        bw = {u: Sg[cat == u].mean(0) for u in np.unique(cat)}
        def st(sel):
            cs_ = cat[sel]; return Sg[sel].mean(0) - np.mean([bw[u] for u in cs_], 0)
        o = st(top); span = pos[-1] - pos[0] + 1; rel = pos[top] - pos[top[0]]; nn = []
        for _ in range(nperm):
            off = rng.integers(1, span); newp = pos[0] + (pos[top[0]] - pos[0] + off + rel) % span; sel = np.searchsorted(pos, newp).clip(0, n - 1)
            sel = np.where(np.abs(pos[sel] - newp) <= np.abs(pos[(sel - 1).clip(0)] - newp), sel, (sel - 1).clip(0)); nn.append(st(sel))
        nn = np.array(nn); ob.append(o); nl.append(nn); lz.append((o - nn.mean(0)) / (nn.std(0) + 1e-12))
    return np.array(ob), np.array(nl), np.array(lz)
ob, nl, lz = tilt()
for j, c in enumerate(CN):
    o = ob[:, j].mean(); nn = nl[:, :, j].mean(0); P(f"    {c:6s} Δ {o:+.4f} (귀무 {nn.mean():+.4f}) 양측 p {2 * min((1 + (nn >= o).sum()), (1 + (nn <= o).sum())) / (len(nn) + 1):.3f}")
# ---------- E7 ----------
P("\n=== E7: GTEx v8 eQTL·sQTL (P0 CS 변이 PIP ≥ 0.1; 로지스틱, 표준화 계수) ===")
G = pd.read_parquet(X/"annot"/"gtex.parquet").merge(D[["atlas_variant", "S_발현", "S_스플라이싱", "maf", "log_tss"]], on="atlas_variant")
G["Ye"], G["Ys"] = (G.n_eqtl > 0).astype(float), (G.n_sqtl > 0).astype(float); P(f"  변이 {len(G)}: eQTL 있음 {G.Ye.mean():.0%}, sQTL 있음 {G.Ys.mean():.0%}")
Z = G[["S_스플라이싱", "S_발현", "maf", "log_tss"]].apply(lambda v: (v - v.mean()) / v.std()); Xd = sm.add_constant(Z)
for y, main, cross in [("Ys", "S_스플라이싱", "S_발현"), ("Ye", "S_발현", "S_스플라이싱")]:
    try:
        f = sm.Logit(G[y], Xd).fit(disp=0); pm = stats.norm.sf(f.tvalues[main])
        P(f"  {y}: {main[2:]} OR {np.exp(f.params[main]):.2f}/SD (단측 p {pm:.4f}) → {'지지' if pm < .05 else '불지지'} | 교차 {cross[2:]} OR {np.exp(f.params[cross]):.2f} (양측 p {f.pvalues[cross]:.3f})")
    except Exception as e: P(f"  {y}: 적합 실패 ({type(e).__name__})")
# ---------- 좌위 목록 ----------
LT = pd.DataFrame({"locus": D.locus.unique()}); LT[[f"z_tilt_{c}" for c in CN]] = lz
top = CS.sort_values("pip").groupby("locus").tail(1).set_index("locus")[["lead", "pip"] + [f"lead_{c}" for c in CN]]
gl = G.groupby("locus").agg(eqtl_genes=("eqtl_genes", lambda s: ";".join(sorted({x for v in s for x in v.split(";") if x}))), sqtl_genes=("sqtl_genes", lambda s: ";".join(sorted({x for v in s for x in v.split(";") if x}))))
LT = LT.set_index("locus").join(top).join(gl); LT["profile"] = np.select([(LT["z_tilt_스플라이싱"] > 1.64) & (LT["z_tilt_발현"] > 1.64), LT["z_tilt_스플라이싱"] > 1.64, LT["z_tilt_발현"] > 1.64], ["발현+스플라이싱", "스플라이싱형", "발현형"], "뚜렷하지 않음")
LT.to_csv(RES/"e05_locus_table.csv"); P("\n좌위 목록(기술): " + ", ".join(f"{k} {v}" for k, v in LT.profile.value_counts().items()))
(RES/"e05_tests_1e2.txt").write_text("\n".join(log), encoding="utf-8")
