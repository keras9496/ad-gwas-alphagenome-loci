"""1단계-1: Bellenguez 2022 1단계 요약통계에서 유전자좌 정의 (PROTOCOL_stage1 규칙).
 - 표본 수 ≥ 최대의 80 %, 이중대립 SNV, MAF ≥ 1 %
 - p < 5e-8 선도 SNP 거리 기반(±1 Mb) 탐욕 선택, 유전자좌 = 선도 ±250 kb, APOE(chr19:43.9–45.9 Mb)·MHC(chr6:25–34 Mb) 제외
출력: results/loci.csv, D:/AD_GWAS_Atlas_data/work/locus_snps.parquet, work/loci_ranges.txt (plink2 --extract range)
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT/"results"; RES.mkdir(exist_ok=True)
DATA = Path("D:/AD_GWAS_Atlas_data"); WORK = DATA/"work"; WORK.mkdir(exist_ok=True)
cols = ["variant_id", "p_value", "chromosome", "base_pair_location", "effect_allele", "other_allele", "effect_allele_frequency", "beta", "standard_error", "n_cases", "n_controls", "variant_alternate_id"]
S = pd.read_csv(DATA/"sumstats"/"GCST90027158_buildGRCh38.tsv.gz", sep="\t", usecols=cols, dtype={"chromosome": str})
n0 = len(S); S["N"] = S.n_cases + S.n_controls
S = S[(S.effect_allele.str.len() == 1) & (S.other_allele.str.len() == 1)]
S = S[S.N >= 0.8 * S.N.max()]
S["maf"] = np.minimum(S.effect_allele_frequency, 1 - S.effect_allele_frequency); S = S[S.maf >= 0.01]
print(f"요약통계 {n0:,}행 → SNV·N≥80%·MAF≥1% {len(S):,}행 (N 최대 {S.N.max():,})")
S["chr"] = S.chromosome; S["pos"] = S.base_pair_location
def excluded(c, p): return (c == "19" and 43_900_000 <= p <= 45_900_000) or (c == "6" and 25_000_000 <= p <= 34_000_000)
sig = S[S.p_value < 5e-8].sort_values("p_value")
leads, taken = [], []
for r in sig.itertuples(index=False):
    if excluded(r.chr, r.pos): continue
    if any(c == r.chr and abs(p - r.pos) <= 1_000_000 for c, p in taken): continue
    taken.append((r.chr, r.pos)); leads.append(r)
L = pd.DataFrame(leads)[["variant_id", "chr", "pos", "p_value", "beta", "maf", "N", "variant_alternate_id"]]
L["locus"] = [f"L{i+1:03d}_chr{c}_{p//1000}k" for i, (c, p) in enumerate(zip(L.chr, L.pos))]
L["start"], L["end"] = L.pos - 250_000, L.pos + 250_000
print(f"유전자좌 {len(L)}개 (APOE·MHC 제외), 최소 p {L.p_value.min():.1e}, 최대 p {L.p_value.max():.1e}")
parts = []
for r in L.itertuples(index=False):
    s = S[(S.chr == r.chr) & (S.pos >= r.start) & (S.pos <= r.end)].copy(); s["locus"] = r.locus; s["lead"] = r.variant_id; parts.append(s)
X = pd.concat(parts, ignore_index=True); X["chi2"] = (X.beta / X.standard_error) ** 2
sp = X.variant_alternate_id.str.split(":", expand=True)            # chrN, pos, REF, ALT (hg38)
X["ref"], X["alt"] = sp[2], sp[3]
X["atlas_variant"] = sp[0] + ":" + sp[1] + ":" + sp[2] + ">" + sp[3]
X.to_parquet(WORK/"locus_snps.parquet", index=False)
L = L.merge(X.groupby("locus").size().rename("n_snp").reset_index(), on="locus"); L.to_csv(RES/"loci.csv", index=False)
with open(WORK/"loci_ranges.txt", "w") as f:
    for r in L.itertuples(index=False): f.write(f"{r.chr}\t{r.start}\t{r.end}\t{r.locus}\n")
print(f"유전자좌 SNP 합계 {len(X):,} (유전자좌당 중앙 {int(L.n_snp.median())})"); print(X.atlas_variant.head(3).tolist())
print(L[["locus", "variant_id", "p_value", "n_snp"]].to_string(index=False))
