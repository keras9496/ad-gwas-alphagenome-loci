"""실험 1e e14: 원고 R34 대응 — F1 부류 무관 지표(생물 시료 일치, 트랙 순위 상관)의 측정치 × LD 층 표 (2026-10-05).
 e07 F1과 같은 자료·상위 집합(좌위별 χ² 상위 1 %, 최소 5개)·귀무(거리 보존 평행이동)·좌위 평균 통계량을 쓴다.
 - e07이 이미 보고한 칸: 관측값과 쌍 수를 다시 계산해 e07 출력과 대조한다(재현 확인).
 - e07이 계산하지 않은 칸(생물 시료 일치의 낮은·높은 LD 층, 트랙 순위의 높은 LD 층): **사후** 계산이다. 귀무 1,000회, 시드 20261005.
 - LD 층 정의는 e07과 같다: 상위 집합 쌍의 r² < 0.1(낮음), ≥ 0.5(높음). e07처럼 귀무 집합의 쌍에도 관측 쌍의 LD 층 표시를 그대로 쓴다.
출력: results/e14_f1_strata/f1_strata_table.csv, e14_f1_strata.txt
"""
import warnings; warnings.filterwarnings("ignore")
import re, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
EXP = Path(__file__).resolve().parents[1]; OUT = EXP/"results"/"e14_f1_strata"; OUT.mkdir(exist_ok=True)
W = Path("D:/AD_GWAS_Atlas_data/work"); C = Path("D:/AD_GWAS_Atlas_data/exp1e")/"ct"
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
# 세포 부류 규칙은 e07에서 그대로 가져온다(소스에서 CLASS_RULES 블록만 실행)
src = (EXP/"scripts"/"e07_celltype_tests.py").read_text(encoding="utf-8")
ns = {"re": re}; exec(src[src.index("CLASS_RULES = ["):src.index("TR = pd.read_csv")], ns); cls = ns["cls"]
TR = pd.read_csv(C/"tracks_RNA_SEQ.csv"); TR["cls"] = TR.biosample_name.astype(str).map(cls)
TS = pd.read_csv(C/"tracks_SPLICE_JUNCTIONS.csv"); TS["cls"] = TS.biosample_name.astype(str).map(cls)
CL = sorted(TR.cls.unique())
# ---- RNA 자료 (e07 33–42행과 같음)
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2"]
D0 = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
Z = [np.load(f, allow_pickle=True) for f in sorted(C.glob("rna_*.npz"))]; Qv = np.vstack([z["Q"] for z in Z]).astype(np.float32); vid = np.concatenate([z["variant"] for z in Z])
M = pd.concat([pd.read_parquet(f) for f in sorted(C.glob("rna_meta_*.parquet"))]).drop_duplicates("atlas_variant").set_index("atlas_variant")
D = D0[D0.atlas_variant.isin(M.index[M.tr.notna()])].sort_values(["locus", "pos"]).reset_index(drop=True)
D["tr"] = M.loc[D.atlas_variant, "tr"].astype(int).to_numpy(); D["bios"] = TR.biosample_name.to_numpy()[D.tr]; D["cls"] = TR.cls.to_numpy()[D.tr]
D["ci"] = D.cls.map({c: i for i, c in enumerate(CL)})
rowpos = pd.Series(np.arange(len(vid)), index=vid); RK = stats.rankdata(np.nan_to_num(Qv[rowpos.loc[D.atlas_variant].to_numpy()]), axis=1).astype(np.float32)
RK = (RK - RK.mean(1, keepdims=True)) / (RK.std(1, keepdims=True) + 1e-9)
# ---- 스플라이싱 자료 (e07 95–110행과 같음)
S = pd.concat([pd.read_parquet(f) for f in sorted(C.glob("spl_*.parquet")) if f.stat().st_size > 0 and len(pd.read_parquet(f))]).drop_duplicates("atlas_variant")
cols = [c for c in S.columns if c not in ("atlas_variant", "gene")]; S["tr"] = np.nanargmax(np.nan_to_num(S[cols].astype(float).to_numpy(), nan=-1), 1)
Sfull = S.set_index("atlas_variant")
DS = D.drop(columns=["tr", "bios", "cls", "ci"]).merge(S[["atlas_variant", "tr"]], on="atlas_variant").sort_values(["locus", "pos"]).reset_index(drop=True)
DS["bios"] = TS.biosample_name.to_numpy()[DS.tr]; DS["cls"] = TS.cls.to_numpy()[DS.tr]; DS["ci"] = DS.cls.map({c: i for i, c in enumerate(CL)}).fillna(-1).astype(int)
RS = stats.rankdata(np.nan_to_num(Sfull.loc[DS.atlas_variant, cols].astype(float).to_numpy(), nan=0), axis=1).astype(np.float32); RS = (RS - RS.mean(1, keepdims=True)) / (RS.std(1, keepdims=True) + 1e-9)
P(f"RNA SNV {len(D):,}, 트랙 {RK.shape[1]} | 스플라이싱 SNV {len(DS):,}, 트랙 {RS.shape[1]}")
LDC = {}
def ld_r2(lc, av):
    if lc not in LDC: z = np.load(W/"ld"/f"{lc}.npz", allow_pickle=True); LDC[lc] = (pd.Index(z["variant"].astype(str)), z["R"].astype(np.float32))
    v, R_ = LDC[lc]; ix = v.get_indexer(av); out = np.full((len(av), len(av)), np.nan); m = ix >= 0; out[np.ix_(m, m)] = R_[np.ix_(ix[m], ix[m])].astype(float) ** 2; return out
MEAS = ["class", "biosample", "track_rank"]; STRATA = ["all", "low_LD", "high_LD"]
def run(D, RK, nperm, rng, frac=.01):
    ob, nl, npair = [], [], []
    for lc, g in D.groupby("locus", sort=False):
        idx = g.index.to_numpy(); pos = g.pos.to_numpy(); n = len(idx); k = max(5, int(round(frac * n)))
        top = np.sort(np.argsort(-g.chi2.to_numpy())[:k]); iu = np.triu_indices(k, 1); r2 = ld_r2(lc, g.atlas_variant.to_numpy()[top])
        msk = {"all": np.ones(len(iu[0]), bool), "low_LD": r2[iu] < .1, "high_LD": r2[iu] >= .5}
        ci, bs = g.ci.to_numpy(), g.bios.to_numpy()
        def st(sel):
            v = {"class": (ci[sel][iu[0]] == ci[sel][iu[1]]).astype(float), "biosample": (bs[sel][iu[0]] == bs[sel][iu[1]]).astype(float),
                 "track_rank": ((lambda R_: (R_ @ R_.T / R_.shape[1])[iu])(RK[idx[sel]]))}
            return np.array([[v[m][msk[s]].mean() if msk[s].any() else np.nan for s in STRATA] for m in MEAS])
        o = st(top); span = pos[-1] - pos[0] + 1; rel = pos[top] - pos[top[0]]; nul = []
        for _ in range(nperm):
            off = rng.integers(1, span); newp = pos[0] + (pos[top[0]] - pos[0] + off + rel) % span; sel = np.searchsorted(pos, newp).clip(0, n - 1)
            sel = np.where(np.abs(pos[sel] - newp) <= np.abs(pos[(sel - 1).clip(0)] - newp), sel, (sel - 1).clip(0)); nul.append(st(sel))
        ob.append(o); nl.append(np.array(nul)); npair.append([msk[s].sum() for s in STRATA])
    ob, nl, npair = np.array(ob), np.array(nl), np.array(npair); rows = []
    for i, m in enumerate(MEAS):
        for j, s in enumerate(STRATA):
            ok = ~np.isnan(ob[:, i, j]); o_ = ob[ok, i, j].mean(); Nn = np.nanmean(nl[ok, :, i, j], 0)
            rows.append(dict(measure=m, ld_stratum=s, observed=o_, null_mean=np.nanmean(Nn), p_one_sided=(1 + (Nn >= o_).sum()) / (len(Nn) + 1),
                             n_loci=int(ok.sum()), n_pairs=int(npair[ok, j].sum())))
    return pd.DataFrame(rows)
out = []
for lab, d, rk in [("RNA-seq (371 tracks)", D, RK), ("splicing (367 tracks)", DS, RS)]:
    T = run(d, rk, 1000, np.random.default_rng(20261005)); T.insert(0, "prediction", lab); out.append(T)
T = pd.concat(out, ignore_index=True)
# e07이 보고한 칸 표시와 대조
E7 = {("class", "all"), ("class", "low_LD"), ("class", "high_LD"), ("biosample", "all"), ("track_rank", "all"), ("track_rank", "low_LD")}
T["in_e07"] = [("yes (pre-specified secondary)" if (r.measure, r.ld_stratum) in E7 else "no (post hoc, e14)") for r in T.itertuples()]
T.to_csv(OUT/"f1_strata_table.csv", index=False)
P("\n=== 측정치 × LD 층 (좌위 평균; 귀무 = 거리 보존 평행이동 1,000회, 단측 P = 관측 이상 비율) ===")
for r in T.itertuples():
    P(f"  {r.prediction:22s} {r.measure:10s} {r.ld_stratum:7s} 관측 {r.observed:.4f} | 귀무 {r.null_mean:.4f} | P {r.p_one_sided:.3f} | 좌위 {r.n_loci} | 쌍 {r.n_pairs:,} | {r.in_e07}")
(OUT/"e14_f1_strata.txt").write_text("\n".join(log), encoding="utf-8")
