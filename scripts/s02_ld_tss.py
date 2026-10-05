# [public version] Ensembl REST lookup of TSS positions replaced by a read of the cached tss_cache.json; everything else as run.
"""1단계-2: 유전자좌 SNP 에 1000G EUR(비친족 525명) LD 점수, 선도 SNP 와의 r², 최근접 protein-coding TSS 거리를 붙인다.
 LD 점수 = 유전자좌 안(±250 kb) 참조 SNP 와의 r² 합 (편향 보정 r² − (1−r²)/(n−2)).
 매칭: 요약통계 chr:pos:REF>ALT ↔ 1000G chr:pos:ref:alt (대립 순서 뒤집힘 허용).
사용: python s02_ld_tss.py [locus ...]   (인자 없으면 전체)
출력: D:/AD_GWAS_Atlas_data/work/ld/<locus>.npz (r 행렬, 2단계 SuSiE 용), work/locus_snps_annot.parquet
"""
import sys, json, subprocess
import numpy as np, pandas as pd
from pathlib import Path
DATA = Path("D:/AD_GWAS_Atlas_data"); WORK = DATA/"work"; (WORK/"ld").mkdir(exist_ok=True)
PL = str(DATA/"tools"/"plink2.exe")
X = pd.read_parquet(WORK/"locus_snps.parquet"); R = pd.read_csv(Path(__file__).resolve().parents[1]/"results"/"loci.csv", dtype={"chr": str})
loci = sys.argv[1:] or R.locus.tolist()

TSS_CACHE = WORK/"tss_cache.json"; TC = json.loads(TSS_CACHE.read_text()) if TSS_CACHE.exists() else {}
def tss(chrom, lo, hi):
    k = f"{chrom}:{max(lo,1)}-{hi}"
    if k not in TC: raise KeyError(f"{k} not in tss_cache.json: provide Ensembl protein-coding gene TSS positions for this window (see README)")
    return np.array(TC[k] or [np.nan], float)

out = []
for lc in loci:
    r = R[R.locus == lc].iloc[0]; s = X[X.locus == lc].copy()
    pre = WORK/"ld"/f"{lc}_tmp"
    subprocess.run([PL, "--pfile", str(WORK/"loci_eur"), "--chr", str(r.chr), "--from-bp", str(r.start), "--to-bp", str(r.end),
                    "--export", "A-transpose", "--out", str(pre), "--silent"], check=True)
    G = pd.read_csv(f"{pre}.traw", sep="\t"); ids = G.SNP.to_numpy(); g = G.iloc[:, 6:].to_numpy(float)
    for f in Path(WORK/"ld").glob(f"{lc}_tmp*"): f.unlink()
    g = np.where(np.isnan(g), np.nanmean(g, 1, keepdims=True), g); ok = g.std(1) > 0; g, ids = g[ok], ids[ok]
    z = (g - g.mean(1, keepdims=True)) / g.std(1, keepdims=True); n = z.shape[1]
    Rm = z @ z.T / n; r2 = Rm ** 2; r2a = r2 - (1 - r2) / (n - 2)
    ldscore = r2a.sum(1)
    # 요약통계 SNP 매칭 (REF/ALT 뒤집힘 허용)
    key = {}
    for i, v in enumerate(ids):
        c, p, a1, a2 = v.split(":"); key[(p, a1, a2)] = (i, 1); key[(p, a2, a1)] = (i, -1)
    m = [key.get((str(p), a, b), (-1, 0)) for p, a, b in zip(s.pos, s.ref, s.alt)]
    s["ref_idx"] = [i for i, _ in m]; s["ref_flip"] = [f for _, f in m]
    s = s[s.ref_idx >= 0].copy()
    s["ldscore"] = ldscore[s.ref_idx]
    lead = s.loc[s.p_value.idxmin()]; s["r2_lead"] = r2[s.ref_idx, int(lead.ref_idx)]
    t = tss(r.chr, int(r.start) - 1_000_000, int(r.end) + 1_000_000); s["tss_dist"] = np.nanmin(np.abs(s.pos.to_numpy()[:, None] - t[None, :]), 1)
    sub = Rm[np.ix_(s.ref_idx, s.ref_idx)] * np.outer(s.ref_flip, s.ref_flip)       # 요약통계 ALT 기준 부호의 r
    np.savez_compressed(WORK/"ld"/f"{lc}.npz", R=sub.astype(np.float32), variant=s.atlas_variant.to_numpy())
    out.append(s.drop(columns=["ref_idx"])); print(f"{lc}: 요약통계 {int((X.locus == lc).sum())} → 1000G 매칭 {len(s)}, 선도 {lead.variant_id} (r2_lead≥0.8: {int((s.r2_lead >= .8).sum())})", flush=True)
A = pd.concat(out, ignore_index=True)
f = WORK/"locus_snps_annot.parquet"
if f.exists() and sys.argv[1:]:
    old = pd.read_parquet(f); A = pd.concat([old[~old.locus.isin(loci)], A], ignore_index=True)
A.to_parquet(f, index=False)
