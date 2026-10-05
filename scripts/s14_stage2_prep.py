"""2단계 준비: 유전자좌별 z(ALT 기준), LD(R, float32 바이너리), Atlas 맥락 백분위(S_c), 세포형 중립 F.
출력: D:/AD_GWAS_Atlas_data/work/s2/<locus>_R.bin, <locus>_z.csv, work/s2/snv_profile.parquet
"""
import numpy as np, pandas as pd
from pathlib import Path
WORK = Path("D:/AD_GWAS_Atlas_data/work"); OUT = WORK/"s2"; OUT.mkdir(exist_ok=True)
MODS = ["DNASE", "H3K27ac", "H3K4me1", "CAGE"]; CTX = ["myeloid", "neuron", "astrocyte", "liver", "nonliver"]
X = pd.read_parquet(WORK/"locus_snps_annot.parquet")
A = pd.concat([pd.read_parquet(f) for f in sorted((WORK/"atlas").glob("*.parquet"))], ignore_index=True)
C = pd.concat([pd.read_parquet(f) for f in sorted((WORK/"atlas_s3").glob("*.parquet"))], ignore_index=True)
AC = A.merge(C, on=["atlas_variant", "locus"], how="outer")
for c in CTX: AC[f"S_{c}"] = np.mean([AC[f"{m}_{c}"].fillna(0).rank(pct=True).to_numpy() for m in MODS], 0)
AC["F"] = AC[[f"S_{c}" for c in CTX]].mean(1)
X["z"] = X.beta / X.standard_error * np.where(X.effect_allele == X.alt, 1.0, -1.0)     # ALT 기준
X["beta_alt"] = X.beta * np.where(X.effect_allele == X.alt, 1.0, -1.0)
prof = []
for lc, g in X.groupby("locus", sort=False):
    d = np.load(WORK/"ld"/f"{lc}.npz", allow_pickle=True); v = d["variant"].astype(str)
    g = g.set_index("atlas_variant").loc[v].reset_index(); assert len(g) == len(v)
    d["R"].astype(np.float32).tofile(OUT/f"{lc}_R.bin")
    g[["atlas_variant", "z"]].assign(n=int(g.N.median())).to_csv(OUT/f"{lc}_z.csv", index=False)
    prof.append(g[["locus", "atlas_variant", "variant_id", "chr", "pos", "ref", "alt", "effect_allele", "beta_alt", "z", "p_value", "maf"]])
P = pd.concat(prof, ignore_index=True).merge(AC[["atlas_variant", "locus", "F"] + [f"S_{c}" for c in CTX]], on=["atlas_variant", "locus"], how="left")
P.to_parquet(OUT/"snv_profile.parquet", index=False)
print(f"유전자좌 {P.locus.nunique()}, SNV {len(P):,}, Atlas 누락 {int(P.F.isna().sum())}")
