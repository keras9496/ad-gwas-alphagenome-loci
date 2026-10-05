"""실험 1e-8 f04: E8 검정 (PROTOCOL_exp1e8_2026-10-05).
 주: 집합 A(76 좌위)에서 발현·스플라이싱 ρ̄ > 0 (각 단측 α 0.025; e05 E5a와 같은 rho_bar, 원형 이동 1,000회, 시드 20261006).
 부: E8-S1 위치 보정(VEP 결과 유형 + log 엑손 거리; 기준 같은 방향 + 단측 P < 0.05).
 기술: 나머지 종류, χ² 10분위, 원래 75 + A 합산, 집합 B.
 종류 점수 백분위는 1e-2 참조 분포(75개 영역 SNV 111,446개)의 경험 누적분포(동점 평균 순위)로 매긴다.
 E8_DRYRUN=1: 좌위 안 χ²를 무작위로 섞은 자료로 형식만 확인한다(결과 아님; results/e8/dryrun.txt).
출력: results/e8/e8_tests.txt, e8_locus_rho.csv
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"/"e8"; RES.mkdir(parents=True, exist_ok=True)
W = Path("D:/AD_GWAS_Atlas_data/work"); X1 = Path("D:/AD_GWAS_Atlas_data/exp1e"); X8 = Path("D:/AD_GWAS_Atlas_data/exp1e8")
DRY = bool(os.environ.get("E8_DRYRUN")); log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2", "maf", "ldscore", "tss_dist"]
FEAT_ABS = ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION", "DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"]
CLS = {"발현": ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION"], "조절": ["DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"], "스플라이싱": ["MERGED_SPLICING"],
       "단백질": ["ALPHAMISSENSE", "PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"], "보존": ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]}
CN = list(CLS)
def raw(D):   # e05와 같은 원 특성 변환(백분위 전 값)
    R = {}
    for f in FEAT_ABS: R[f] = D[f"mf_MAX_ABS_{f}"].astype(float).abs().fillna(0).to_numpy()
    for f in ["MERGED_SPLICING", "ALPHAMISSENSE"]: R[f] = D[f"mf_{f}"].astype(float).abs().fillna(0).to_numpy()
    return R
# ---- 참조 분포 (1e-2와 같은 75개 영역 자료)
D0 = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
MF0 = pd.concat([pd.read_parquet(f) for f in sorted((X1/"mf").glob("part_*.parquet"))]).drop_duplicates("atlas_variant")
D0 = D0.merge(MF0, on="atlas_variant"); assert len(D0) == 111_446, len(D0)
R0 = raw(D0); cons_min = {f: D0[f"mf_{f}"].min() for f in ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]}
for f in cons_min: R0[f] = D0[f"mf_{f}"].astype(float).fillna(cons_min[f]).to_numpy()
REFS = {f: np.sort(v) for f, v in R0.items()}
def ecdf(f, v):   # pandas rank(pct=True, method="average")와 같은 정의: (#< + #≤ + 1) / 2n
    r = REFS[f]; lo = np.searchsorted(r, v, "left"); hi = np.searchsorted(r, v, "right"); return (lo + hi + 1) / 2 / len(r)
def scores(D):
    R = raw(D)
    for f in cons_min: R[f] = D[f"mf_{f}"].astype(float).fillna(cons_min[f]).to_numpy()
    F = {f: ecdf(f, R[f]) for f in R}
    for f in ["PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"]: F[f] = (D[f"mf_{f}"].astype(float).fillna(0) > 0).astype(float).to_numpy()
    for c in CN: D[f"S_{c}"] = np.max(np.column_stack([F[f] for f in CLS[c]]), 1)
    return D
# 참조 자료에 같은 변환을 적용하면 1e-2 점수가 그대로 나와야 한다(척도 고정 확인)
D0 = scores(D0); chk = D0.copy()
for f in R0: chk[f"_p_{f}"] = pd.Series(R0[f]).rank(pct=True).to_numpy()
assert np.allclose(chk[[f"_p_{f}" for f in ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION"]]].max(1), D0.S_발현), "참조 척도 불일치"
P("참조 척도 확인: 1e-2 종류 점수와 같다")
# ---- E8 자료
D = pd.read_parquet(X8/"snps_annot.parquet")[K + ["set"]]
MF = pd.concat([pd.read_parquet(f) for f in sorted((X8/"mf").glob("part_*.parquet"))]).drop_duplicates("atlas_variant")
n_all = len(D); D = D.merge(MF, on="atlas_variant"); D = D[D[[c for c in MF.columns if c.startswith("mf_")]].notna().any(axis=1)]
D = D.merge(pd.read_parquet(X8/"annot"/"vep.parquet"), on="atlas_variant", how="left").merge(pd.read_parquet(X8/"annot"/"exon_dist.parquet"), on="atlas_variant", how="left")
D = scores(D).sort_values(["locus", "pos"]).reset_index(drop=True); D["log_tss"] = np.log10(D.tss_dist + 1); D["log_exon"] = np.log10(D.exon_dist.fillna(1e6) + 1)
D0["log_tss"] = np.log10(D0.tss_dist + 1); D0["set"] = "orig75"
P(f"E8 SNV {len(D):,}/{n_all:,} (Atlas 특성 있음) | A 좌위 {D[D.set == 'A'].locus.nunique()} SNV {int((D.set == 'A').sum()):,} | B 좌위 {D[D.set == 'B'].locus.nunique()} SNV {int((D.set == 'B').sum()):,}")
if DRY:
    rs = np.random.default_rng(1); D["chi2"] = D.groupby("locus").chi2.transform(lambda v: rs.permutation(v.to_numpy())); P("*** 시험 실행: 좌위 안 χ²를 섞었다. 결과 아님 ***")
# ---- ρ̄ (e05와 같은 코드)
rng = np.random.default_rng(20261006)
def rres(v, Z):
    rv = stats.rankdata(v); Zr = np.column_stack([np.ones(len(v))] + [stats.rankdata(z) for z in Z.T]); return rv - Zr @ np.linalg.lstsq(Zr, rv, rcond=None)[0]
def rho_bar(G, col, conf, nperm=1000):
    obs, nul, loc = [], [], []
    for lc, g in G.groupby("locus", sort=False):
        Z = g[conf].to_numpy(float); Z = Z[:, Z.std(0) > 0]; y = rres(g.chi2.to_numpy(), Z); s = g[col].to_numpy(float); n = len(s)
        obs.append(np.corrcoef(rres(s, Z), y)[0, 1]); loc.append(lc); sh = rng.integers(max(1, n // 20), n - max(1, n // 20), nperm); nul.append([np.corrcoef(rres(np.roll(s, k), Z), y)[0, 1] for k in sh])
    o, N = np.array(obs), np.array(nul); nb = N.mean(0)
    return o.mean(), nb, (1 + (nb >= o.mean()).sum()) / (len(nb) + 1), np.mean(o > 0), pd.Series(o, index=loc)
BASE = ["ldscore", "maf", "log_tss"]; A = D[D.set == "A"].copy(); B = D[D.set == "B"].copy()
P("\n=== E8 주 검정: 집합 A (선도 5e-8 ≤ P < 1e-5), 좌위 안 부분 Spearman ρ̄ (사전 방향 > 0, 각 단측 α 0.025) ===")
rows = {}
for c in CN:
    rb, nb, p, fp, per = rho_bar(A, f"S_{c}", BASE); rows[c] = per
    tag = (" → " + ("통과" if p < .025 else "불통과")) if c in ("발현", "스플라이싱") else " (기술)"
    P(f"  {c:6s} ρ̄ {rb:+.4f} (귀무 95 % {np.quantile(nb, .95):+.4f}) 단측 p {p:.4f} | 양의 좌위 {fp:.0%}{tag}")
pd.DataFrame(rows).rename_axis("locus").to_csv(RES/("dryrun_locus_rho.csv" if DRY else "e8_locus_rho.csv"))
P("\n=== E8-S1 (부): 위치 보정 (VEP 결과 유형 + log 엑손 경계 거리; 기준: 같은 방향 + 단측 P < 0.05) ===")
grp = {"coding": ["missense_variant", "synonymous_variant", "stop_gained", "stop_lost", "start_lost", "coding_sequence_variant", "inframe_insertion", "inframe_deletion", "stop_retained_variant", "incomplete_terminal_codon_variant"],
       "utr": ["5_prime_UTR_variant", "3_prime_UTR_variant"], "splice": sorted({"splice_donor_variant", "splice_acceptor_variant", "splice_region_variant", "splice_donor_5th_base_variant", "splice_donor_region_variant", "splice_polypyrimidine_tract_variant"}),
       "intron": ["intron_variant"], "flank": ["upstream_gene_variant", "downstream_gene_variant"], "nc": ["non_coding_transcript_exon_variant", "mature_miRNA_variant"]}
A["cat"] = "other"
for k, vs in grp.items(): A.loc[A.msc.isin(vs), "cat"] = k
A.loc[A.msc == "intergenic_variant", "cat"] = "intergenic"
dm = pd.get_dummies(A.cat, prefix="cat", dtype=float); A = pd.concat([A, dm], axis=1); CONF6 = BASE + ["log_exon"] + [c for c in dm.columns if c != "cat_intergenic"]
P(f"  VEP 있음 {A.msc.notna().mean():.1%} | 결과 유형: " + ", ".join(f"{k} {v:.1%}" for k, v in A.cat.value_counts(normalize=True).items()))
for c in ["발현", "스플라이싱"]:
    rb, nb, p, fp, _ = rho_bar(A, f"S_{c}", CONF6)
    P(f"  {c:6s} ρ̄ {rb:+.4f} 단측 p {p:.4f} → " + ("위치로 설명되지 않음" if (rb > 0 and p < .05) else "위치로 설명될 수 있음"))
P("\n=== 기술: 집합 A 좌위 안 χ² 10분위별 평균 종류 점수 (좌위 평균) ===")
A["dec"] = A.groupby("locus").chi2.transform(lambda v: np.floor(stats.rankdata(v) / (len(v) + 1) * 10)).astype(int)
P("  " + A.groupby(["locus", "dec"])[[f"S_{c}" for c in CN]].mean().groupby("dec").mean().round(3).rename(columns=lambda x: x[2:]).to_string().replace("\n", "\n  "))
if not DRY:
    P("\n=== 기술: 원래 75 + A 합산 (151 좌위) ===")
    PO = pd.concat([D0[K + ["log_tss", "set"] + [f"S_{c}" for c in CN]], A[K + ["log_tss", "set"] + [f"S_{c}" for c in CN]]], ignore_index=True).sort_values(["locus", "pos"]).reset_index(drop=True)
    for c in ["발현", "스플라이싱"]:
        rb, nb, p, fp, _ = rho_bar(PO, f"S_{c}", BASE); P(f"  {c:6s} ρ̄ {rb:+.4f} 단측 p {p:.4f} | 양의 좌위 {fp:.0%}")
P("\n=== 기술: 집합 B (선도 1e-5 ≤ P < 5e-5; 용량–반응) ===")
for c in CN:
    rb, nb, p, fp, _ = rho_bar(B, f"S_{c}", BASE); P(f"  {c:6s} ρ̄ {rb:+.4f} (귀무 95 % {np.quantile(nb, .95):+.4f}) 단측 p {p:.4f} | 양의 좌위 {fp:.0%}")
(RES/("dryrun.txt" if DRY else "e8_tests.txt")).write_text("\n".join(log), encoding="utf-8")
print("f04 완료", flush=True)
