# [public version] GWAS Catalog and Atlas query blocks removed; Ensembl TSS lookup replaced by a read of tss_cache.json; LD and SNP preparation as run.
"""재현 25좌위 준비: Catalog 신규 좌위 정의 → SNP(1단계 필터) → LD 점수·TSS(1000G EUR founders) → Atlas 트랙별 |ALT−REF| (156 트랙, trackdecomp 트랙 목록).
출력: D:/AD_GWAS_Atlas_data/work/rep25/{loci.csv, snps_annot.parquet, atlas_tracks.parquet}, results/rep25_loci.csv
"""
import json, subprocess, time
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DATA = Path("D:/AD_GWAS_Atlas_data"); W = DATA/"work"; OUT = W/"rep25"; OUT.mkdir(exist_ok=True)
PL = str(DATA/"tools"/"plink2.exe"); REF = DATA/"ref_1000G"
# 1) Regions: the 25 replication regions were defined from GWAS Catalog associations of GCST90027158 (lead > 1 Mb from the 50 original leads,
#    APOE/MHC excluded, signals within 1 Mb merged, region = lead +/- 250 kb). The Catalog query is not included; provide OUT/loci.csv
#    with columns locus, rsid, chr, pos, cat_p, stageI_p, start, end (Supplementary Table S3 lists the 25 regions).
if not (OUT/"loci.csv").exists(): raise SystemExit("provide rep25/loci.csv (see comment above)")
L = pd.read_csv(OUT/"loci.csv", dtype={"chr": str}); print(f"좌위 {len(L)}", flush=True)
# 2) SNP + LD·TSS
if not (OUT/"snps_annot.parquet").exists():
    cols = ["variant_id", "p_value", "chromosome", "base_pair_location", "effect_allele", "other_allele", "effect_allele_frequency", "beta", "standard_error", "n_cases", "n_controls", "variant_alternate_id"]
    S = pd.read_csv(DATA/"sumstats"/"GCST90027158_buildGRCh38.tsv.gz", sep="\t", usecols=cols, dtype={"chromosome": str}); S["N"] = S.n_cases + S.n_controls
    S = S[(S.effect_allele.str.len() == 1) & (S.other_allele.str.len() == 1) & (S.N >= .8 * S.N.max())]; S["maf"] = np.minimum(S.effect_allele_frequency, 1 - S.effect_allele_frequency); S = S[S.maf >= .01]
    parts = []
    for r in L.itertuples():
        s = S[(S.chromosome == r.chr) & (S.base_pair_location >= r.start) & (S.base_pair_location <= r.end)].copy(); s["locus"] = r.locus; parts.append(s)
    X = pd.concat(parts, ignore_index=True); X["chr"] = X.chromosome; X["pos"] = X.base_pair_location; X["chi2"] = (X.beta / X.standard_error) ** 2
    sp = X.variant_alternate_id.str.split(":", expand=True); X["ref"], X["alt"] = sp[2], sp[3]; X["atlas_variant"] = sp[0] + ":" + sp[1] + ":" + sp[2] + ">" + sp[3]
    L[["chr", "start", "end", "locus"]].to_csv(OUT/"ranges.txt", sep="\t", header=False, index=False)
    subprocess.run([PL, "--pgen", str(REF/"all_hg38.pgen"), "--pvar", str(REF/"all_hg38_noannot.pvar.zst"), "--psam", str(REF/"hg38_corrected.psam"), "--keep", str(W/"eur.keep"),
                    "--extract", "range", str(OUT/"ranges.txt"), "--snps-only", "just-acgt", "--max-alleles", "2", "--maf", "0.005", "--set-all-var-ids", "chr@:#:$r:$a", "--new-id-max-allele-len", "10",
                    "--make-pgen", "--out", str(OUT/"eur"), "--threads", "16", "--silent"], check=True)
    TC_F = W/"tss_cache.json"; TC = json.loads(TC_F.read_text()) if TC_F.exists() else {}
    def tss(c, lo, hi):
        k = f"{c}:{max(lo, 1)}-{hi}"
        if k not in TC: raise KeyError(f"{k} not in tss_cache.json: provide Ensembl protein-coding gene TSS positions for this window (see README)")
        return np.array(TC[k] or [np.nan], float)
    out = []
    for r in L.itertuples():
        s = X[X.locus == r.locus].copy(); pre = OUT/f"{r.locus}_tmp"
        subprocess.run([PL, "--pfile", str(OUT/"eur"), "--chr", r.chr, "--from-bp", str(r.start), "--to-bp", str(r.end), "--export", "A-transpose", "--out", str(pre), "--silent"], check=True)
        G = pd.read_csv(f"{pre}.traw", sep="\t"); ids = G.SNP.to_numpy(); g = G.iloc[:, 6:].to_numpy(float)
        for f in OUT.glob(f"{r.locus}_tmp*"): f.unlink()
        g = np.where(np.isnan(g), np.nanmean(g, 1, keepdims=True), g); ok = g.std(1) > 0; g, ids = g[ok], ids[ok]
        z = (g - g.mean(1, keepdims=True)) / g.std(1, keepdims=True); n = z.shape[1]; Rm = z @ z.T / n; r2 = Rm ** 2; lds = (r2 - (1 - r2) / (n - 2)).sum(1)
        key = {}
        for i, v in enumerate(ids): c_, p_, a1, a2 = v.split(":"); key[(p_, a1, a2)] = i; key[(p_, a2, a1)] = i
        s["ref_idx"] = [key.get((str(p), a, b), -1) for p, a, b in zip(s.pos, s.ref, s.alt)]; s = s[s.ref_idx >= 0].copy(); s["ldscore"] = lds[s.ref_idx]
        lead = s.loc[s.p_value.idxmin()]; s["r2_lead"] = r2[s.ref_idx, int(lead.ref_idx)]
        t = tss(r.chr, r.start - 1_000_000, r.end + 1_000_000); s["tss_dist"] = np.nanmin(np.abs(s.pos.to_numpy()[:, None] - t[None, :]), 1)
        out.append(s.drop(columns=["ref_idx"])); print(f"{r.locus}: {len(s)} SNP, stage I 최소 p {s.p_value.min():.1e}", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(OUT/"snps_annot.parquet", index=False)
X = pd.read_parquet(OUT/"snps_annot.parquet"); print(f"SNP {len(X):,}", flush=True)
# 3) (removed) Track-level Atlas query used only in experiment 1, not in this study.
print("완료")
