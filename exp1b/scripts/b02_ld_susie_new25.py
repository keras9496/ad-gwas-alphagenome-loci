"""실험 1b b02: 신규 25 좌위 LD 행렬(1000G EUR founders, ALT 기준 부호) + SuSiE 입력 → s15_susie.R (P0) 실행.
출력: D:/AD_GWAS_Atlas_data/work/s2_rep25/<locus>_{R.bin,z.csv}, out_P0/
"""
import subprocess, os
import numpy as np, pandas as pd
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; W = Path("D:/AD_GWAS_Atlas_data/work"); O = W/"rep25"; S2 = W/"s2_rep25"; S2.mkdir(exist_ok=True); PL = "D:/AD_GWAS_Atlas_data/tools/plink2.exe"
X = pd.read_parquet(O/"snps_annot.parquet"); L = pd.read_csv(O/"loci.csv", dtype={"chr": str})
X["z"] = X.beta / X.standard_error * np.where(X.effect_allele == X.alt, 1.0, -1.0)
for r in L.itertuples():
    if (S2/f"{r.locus}_z.csv").exists(): continue
    s = X[X.locus == r.locus].copy(); pre = O/f"{r.locus}_ld"
    subprocess.run([PL, "--pfile", str(O/"eur"), "--chr", r.chr, "--from-bp", str(r.start), "--to-bp", str(r.end), "--export", "A-transpose", "--out", str(pre), "--silent"], check=True)
    G = pd.read_csv(f"{pre}.traw", sep="\t"); ids = G.SNP.to_numpy(); g = G.iloc[:, 6:].to_numpy(float)
    for f in O.glob(f"{r.locus}_ld*"): f.unlink()
    g = np.where(np.isnan(g), np.nanmean(g, 1, keepdims=True), g); ok = g.std(1) > 0; g, ids = g[ok], ids[ok]
    zz = (g - g.mean(1, keepdims=True)) / g.std(1, keepdims=True); Rm = zz @ zz.T / zz.shape[1]
    key = {}
    for i, v in enumerate(ids): c_, p_, a1, a2 = v.split(":"); key[(p_, a1, a2)] = (i, 1); key[(p_, a2, a1)] = (i, -1)   # traw COUNTED=첫 대립(REF) 기준 → ALT 기준 부호
    m = [key.get((str(p), a, b), (-1, 0)) for p, a, b in zip(s.pos, s.ref, s.alt)]; s["ix"] = [i for i, _ in m]; s["fl"] = [f for _, f in m]; s = s[s.ix >= 0]
    sub = Rm[np.ix_(s.ix, s.ix)] * np.outer(s.fl, s.fl)
    sub.astype(np.float32).tofile(S2/f"{r.locus}_R.bin"); s[["atlas_variant", "z"]].assign(n=int(s.N.median())).to_csv(S2/f"{r.locus}_z.csv", index=False)
    np.savez_compressed(W/"ld"/f"{r.locus}.npz", R=sub.astype(np.float32), variant=s.atlas_variant.to_numpy())
    print(f"{r.locus}: {len(s)}", flush=True)
src = (EXP.parent/"scripts"/"s15_susie.R").read_text(encoding="utf-8").replace('W <- "D:/AD_GWAS_Atlas_data/work/s2"', 'W <- "D:/AD_GWAS_Atlas_data/work/s2_rep25"')
(S2/"s15_susie_rep25.R").write_text(src, encoding="utf-8")
r = subprocess.run(["Rscript", str(S2/"s15_susie_rep25.R"), "P0"], capture_output=True, text=True, env=dict(os.environ, LANG="en_US.UTF-8"))
print("\n".join(l for l in r.stdout.splitlines() if l.startswith("P0")))
