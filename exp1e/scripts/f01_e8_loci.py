# [public version] Ensembl REST lookup of TSS positions replaced by a read of the cached tss_cache.json; everything else as run.
"""실험 1e-8 f01: E8 좌위 정의 · SNV · 1000G EUR LD 점수와 LD 행렬 · 최근접 TSS 거리 (PROTOCOL_exp1e8_2026-10-05; s01·s02·s31과 같은 규칙).
 선도: 필터 통과 SNV 중 P < 1e-4를 P 순으로 탐욕 선택(±1 Mb). 제외: 선도 P < 5e-8, APOE·MHC, 원래 75 선도에서 1 Mb 이내.
 집합 A: 5e-8 ≤ P < 1e-5 (E001…), 집합 B: 1e-5 ≤ P < 5e-5 (F001…). 창 = 선도 ± 250 kb.
출력: D:/AD_GWAS_Atlas_data/exp1e8/{loci.csv, snps_annot.parquet, ld/<locus>.npz, tss_cache.json}, results/e8/e8_loci.csv
"""
import json, subprocess, time
import numpy as np, pandas as pd
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; SRC = EXP.parent; DATA = Path("D:/AD_GWAS_Atlas_data"); W = DATA/"work"
OUT = DATA/"exp1e8"; (OUT/"ld").mkdir(parents=True, exist_ok=True); RES = EXP/"results"/"e8"; RES.mkdir(parents=True, exist_ok=True)
PL = str(DATA/"tools"/"plink2.exe"); REF = DATA/"ref_1000G"
cols = ["variant_id", "p_value", "chromosome", "base_pair_location", "effect_allele", "other_allele", "effect_allele_frequency", "beta", "standard_error", "n_cases", "n_controls", "variant_alternate_id"]
S = pd.read_csv(DATA/"sumstats"/"GCST90027158_buildGRCh38.tsv.gz", sep="\t", usecols=cols, dtype={"chromosome": str}); S["N"] = S.n_cases + S.n_controls
S = S[(S.effect_allele.str.len() == 1) & (S.other_allele.str.len() == 1)]; S = S[S.N >= .8 * S.N.max()]
S["maf"] = np.minimum(S.effect_allele_frequency, 1 - S.effect_allele_frequency); S = S[S.maf >= .01]
S["chr"] = S.chromosome; S["pos"] = S.base_pair_location
print(f"필터 후 SNV {len(S):,}", flush=True)
# ---- 1) 좌위
if not (OUT/"loci.csv").exists():
    L0 = pd.read_csv(SRC/"results"/"loci.csv", dtype={"chr": str}); R0 = pd.read_csv(W/"rep25"/"loci.csv", dtype={"chr": str})
    old = pd.concat([L0[["chr", "pos"]], R0[["chr", "pos"]]]); assert len(old) == 75
    def excluded(c, p): return (c == "19" and 43_900_000 <= p <= 45_900_000) or (c == "6" and 25_000_000 <= p <= 34_000_000)
    cand = S[S.p_value < 1e-4].sort_values("p_value"); taken, leads = {}, []
    for r in cand.itertuples(index=False):
        t = taken.setdefault(r.chr, [])
        if any(abs(q - r.pos) <= 1_000_000 for q in t): continue
        t.append(r.pos); leads.append(r)
    Ld = pd.DataFrame(leads)
    keep = [(p >= 5e-8) and not excluded(c, q) and not ((old.chr == c) & ((old.pos - q).abs() <= 1_000_000)).any() for c, q, p in zip(Ld.chr, Ld.pos, Ld.p_value)]
    Ld = Ld[keep].copy()
    Ld["set"] = np.where(Ld.p_value < 1e-5, "A", np.where(Ld.p_value < 5e-5, "B", "")); Ld = Ld[Ld.set != ""].sort_values("p_value").reset_index(drop=True)
    Ld["locus"] = ""
    for s_, pre in [("A", "E"), ("B", "F")]:
        ix = Ld.index[Ld.set == s_]; Ld.loc[ix, "locus"] = [f"{pre}{i + 1:03d}_chr{c}_{p // 1000}k" for i, (c, p) in enumerate(zip(Ld.loc[ix, "chr"], Ld.loc[ix, "pos"]))]
    Ld["start"], Ld["end"] = Ld.pos - 250_000, Ld.pos + 250_000
    Ld = Ld.rename(columns={"variant_id": "rsid"})[["locus", "set", "rsid", "chr", "pos", "p_value", "beta", "maf", "start", "end", "variant_alternate_id"]]
    Ld.to_csv(OUT/"loci.csv", index=False)
L = pd.read_csv(OUT/"loci.csv", dtype={"chr": str}); print(f"좌위 A {int((L.set == 'A').sum())}, B {int((L.set == 'B').sum())}", flush=True)
assert (L.set == "A").sum() == 76 and (L.set == "B").sum() == 159, "프로토콜의 좌위 수와 다르다"
# ---- 2) SNV + LD + TSS
if not (OUT/"snps_annot.parquet").exists():
    parts = []
    for r in L.itertuples():
        s = S[(S.chr == r.chr) & (S.pos >= r.start) & (S.pos <= r.end)].copy(); s["locus"] = r.locus; s["set"] = r.set; s["lead"] = r.rsid; parts.append(s)
    X = pd.concat(parts, ignore_index=True); X["chi2"] = (X.beta / X.standard_error) ** 2
    sp = X.variant_alternate_id.str.split(":", expand=True); X["ref"], X["alt"] = sp[2], sp[3]; X["atlas_variant"] = sp[0] + ":" + sp[1] + ":" + sp[2] + ">" + sp[3]
    L[["chr", "start", "end", "locus"]].to_csv(OUT/"ranges.txt", sep="\t", header=False, index=False)
    if not (OUT/"eur.pgen").exists():
        subprocess.run([PL, "--pgen", str(REF/"all_hg38.pgen"), "--pvar", str(REF/"all_hg38_noannot.pvar.zst"), "--psam", str(REF/"hg38_corrected.psam"), "--keep", str(W/"eur.keep"),
                        "--extract", "range", str(OUT/"ranges.txt"), "--snps-only", "just-acgt", "--max-alleles", "2", "--maf", "0.005", "--set-all-var-ids", "chr@:#:$r:$a",
                        "--new-id-max-allele-len", "10", "--make-pgen", "--out", str(OUT/"eur"), "--threads", "16", "--silent"], check=True)
    TC_F = OUT/"tss_cache.json"; TC = json.loads(TC_F.read_text()) if TC_F.exists() else {}
    def tss(c, lo, hi):
        k = f"{c}:{max(lo, 1)}-{hi}"
        if k not in TC: raise KeyError(f"{k} not in tss_cache.json: provide Ensembl protein-coding gene TSS positions for this window (see README)")
        return np.array(TC[k] or [np.nan], float)
    out = []
    for j, r in enumerate(L.itertuples()):
        s = X[X.locus == r.locus].copy(); pre = OUT/"ld"/f"{r.locus}_tmp"
        subprocess.run([PL, "--pfile", str(OUT/"eur"), "--chr", r.chr, "--from-bp", str(r.start), "--to-bp", str(r.end), "--export", "A-transpose", "--out", str(pre), "--silent"], check=True)
        G = pd.read_csv(f"{pre}.traw", sep="\t"); ids = G.SNP.to_numpy(); g = G.iloc[:, 6:].to_numpy(float)
        for f in (OUT/"ld").glob(f"{r.locus}_tmp*"): f.unlink()
        g = np.where(np.isnan(g), np.nanmean(g, 1, keepdims=True), g); ok = g.std(1) > 0; g, ids = g[ok], ids[ok]
        z = (g - g.mean(1, keepdims=True)) / g.std(1, keepdims=True); n = z.shape[1]; Rm = z @ z.T / n; r2 = Rm ** 2; lds = (r2 - (1 - r2) / (n - 2)).sum(1)
        key = {}
        for i, v in enumerate(ids): c_, p_, a1, a2 = v.split(":"); key[(p_, a1, a2)] = (i, 1); key[(p_, a2, a1)] = (i, -1)
        m = [key.get((str(p), a, b), (-1, 0)) for p, a, b in zip(s.pos, s.ref, s.alt)]
        s["ref_idx"] = [i for i, _ in m]; s["ref_flip"] = [f for _, f in m]; s = s[s.ref_idx >= 0].copy(); s["ldscore"] = lds[s.ref_idx]
        lead = s.loc[s.p_value.idxmin()]; s["r2_lead"] = r2[s.ref_idx, int(lead.ref_idx)]
        t = tss(r.chr, r.start - 1_000_000, r.end + 1_000_000); s["tss_dist"] = np.nanmin(np.abs(s.pos.to_numpy()[:, None] - t[None, :]), 1)
        sub = Rm[np.ix_(s.ref_idx, s.ref_idx)] * np.outer(s.ref_flip, s.ref_flip)
        np.savez_compressed(OUT/"ld"/f"{r.locus}.npz", R=sub.astype(np.float32), variant=s.atlas_variant.to_numpy())
        out.append(s.drop(columns=["ref_idx"]))
        if (j + 1) % 20 == 0 or j + 1 == len(L): print(f"  LD·TSS {j + 1}/{len(L)} ({r.locus}: SNV {len(s)})", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(OUT/"snps_annot.parquet", index=False)
X = pd.read_parquet(OUT/"snps_annot.parquet")
L = L.merge(X.groupby("locus").size().rename("n_snv").reset_index(), on="locus", how="left"); L.to_csv(RES/"e8_loci.csv", index=False)
print(f"SNV A {int((X.set == 'A').sum()):,}, B {int((X.set == 'B').sum()):,} | 좌위당 중앙 {int(L.n_snv.median())}", flush=True)
print("f01 완료")
