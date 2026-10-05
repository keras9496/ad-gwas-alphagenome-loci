"""실험 1e-3 수정 3 — G: fine-mapped 변이의 세포 맥락 엔트로피 (PROTOCOL_exp1e3 수정 3).
 세포 맥락 = 부류별 보정 백분위(부류 안 트랙 max|분위수| 를 전체 SNV 안에서 부류별 백분위) 최대 부류.
 G1(주) PIP 가중 엔트로피 vs CS 묶음 평행이동 귀무 · G2 세포 × 기전 · G3 CS 간 일치 · 민감도.
출력: results/e08_entropy.txt, e08_locus_entropy.csv
"""
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"; W = Path("D:/AD_GWAS_Atlas_data/work"); C = Path("D:/AD_GWAS_Atlas_data/exp1e/ct")
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
src = (EXP/"scripts"/"e07_celltype_tests.py").read_text(encoding="utf-8"); g = {"re": re}; exec(src[src.index("CLASS_RULES = ["):src.index("TR = pd.read_csv")], g); cls = g["cls"]
TR = pd.read_csv(C/"tracks_RNA_SEQ.csv"); TR["cls"] = TR.biosample_name.astype(str).map(cls); TS = pd.read_csv(C/"tracks_SPLICE_JUNCTIONS.csv"); TS["cls"] = TS.biosample_name.astype(str).map(cls)
CL = sorted(TR.cls.unique()); NC = len(CL)
# ---------- 자료 ----------
K = ["atlas_variant", "locus", "pos", "chi2"]
D = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
Z = [np.load(f, allow_pickle=True) for f in sorted(C.glob("rna_*.npz"))]; Qv = np.vstack([z["Q"] for z in Z]).astype(np.float32); vid = np.concatenate([z["variant"] for z in Z])
rp = pd.Series(np.arange(len(vid)), index=vid); D = D[D.atlas_variant.isin(rp.index)].sort_values(["locus", "pos"]).reset_index(drop=True); Q = np.nan_to_num(Qv[rp.loc[D.atlas_variant].to_numpy()])
def calibrate(Qm, tcls, rows_ok=None):
    """부류 점수(부류 안 트랙 최대) → 부류별 백분위(행 = SNV). 반환: 보정 백분위 행렬, 원 최대 행렬."""
    raw = np.column_stack([Qm[:, (tcls == c).to_numpy()].max(1) if (tcls == c).any() else np.zeros(len(Qm)) for c in CL])
    pc = np.column_stack([pd.Series(raw[:, j]).rank(pct=True).to_numpy() for j in range(NC)]); return pc, raw
PC, RAW = calibrate(Q, TR.cls)
def argmax_tiebreak(pc, raw):
    m = pc.max(1, keepdims=True); cand = np.where(pc >= m - 1e-12, raw, -np.inf); return cand.argmax(1)
D["lab"] = argmax_tiebreak(PC, RAW); D["lab_raw"] = TR.cls.map({c: i for i, c in enumerate(CL)}).to_numpy()[Q.argmax(1)]
top3 = np.argsort(-PC, 1)[:, :3]
P(f"SNV {len(D):,}; 보정 세포 맥락 분포: " + ", ".join(f"{CL[k]} {v:.1%}" for k, v in pd.Series(D.lab).value_counts(normalize=True).items()))
P("  (비교) 보정 없는 원 최대 트랙 부류 분포: " + ", ".join(f"{CL[k]} {v:.1%}" for k, v in pd.Series(D.lab_raw).value_counts(normalize=True).head(5).items()))
# 스플라이싱 (G2)
S = pd.concat([pd.read_parquet(f) for f in sorted(C.glob("spl_*.parquet")) if f.stat().st_size > 0 and len(pd.read_parquet(f))]).drop_duplicates("atlas_variant").set_index("atlas_variant")
scol = [c for c in S.columns if c != "gene"]; has_s = D.atlas_variant.isin(S.index).to_numpy()
Qs = np.zeros((len(D), len(scol)), np.float32); Qs[has_s] = np.nan_to_num(S.loc[D.atlas_variant[has_s], scol].astype(float).to_numpy())
PCs = np.zeros((len(D), NC)); RAWs = np.zeros((len(D), NC)); pcs_, raws_ = calibrate(Qs[has_s], TS.cls); PCs[has_s], RAWs[has_s] = pcs_, raws_
bs = argmax_tiebreak(PCs, RAWs); use_s = has_s & (PCs.max(1) > PC.max(1))
D["lab_cm"] = np.where(use_s, NC + bs, D.lab)                       # 0..NC-1 = 발현 × 부류, NC.. = 스플라이싱 × 부류
P(f"스플라이싱 값 보유 {has_s.mean():.1%}; 세포 × 기전에서 스플라이싱이 선택된 SNV {use_s.mean():.1%}")
# ---------- fine-mapping ----------
PIP = pd.concat([pd.read_csv(f).assign(locus=f.name.replace("_pip.csv", "")) for d in [W/"s2"/"out_P0", W/"s2_rep25"/"out_P0"] for f in d.glob("*_pip.csv")])
PIP = PIP[PIP.cs.notna()].drop_duplicates("atlas_variant")
rng = np.random.default_rng(20260929)
def entropy(labels, w, ncat):
    p = np.bincount(labels, weights=w, minlength=ncat); p = p[p > 0] / p.sum(); return float(-(p * np.log(p)).sum())
def entropy_top3(sel_idx, w):
    p = np.zeros(NC)
    for i, wi in zip(sel_idx, w): p[top3[i]] += wi / 3
    p = p[p > 0] / p.sum(); return float(-(p * np.log(p)).sum())
def G(labcol="lab", ncat=NC, pipmin=0.0, mode="hard", nperm=1000, save=False):
    rows, ob, nl = [], [], []
    for lc, gd in D.groupby("locus", sort=False):
        pp = PIP[(PIP.locus == lc) & (PIP.pip >= pipmin)]; pos_ix = pd.Series(np.arange(len(gd)), index=gd.atlas_variant.to_numpy())
        pp = pp[pp.atlas_variant.isin(pos_ix.index)]
        if len(pp) == 0: continue
        w = pp.pip.to_numpy() / pp.pip.sum(); neff = 1 / (w ** 2).sum()
        if neff < 2: continue
        sel = pos_ix.loc[pp.atlas_variant].to_numpy(); gi = gd.index.to_numpy(); lab = gd[labcol].to_numpy().astype(int); pos = gd.pos.to_numpy(); n = len(gd)
        H = (lambda s: entropy(lab[s], w, ncat)) if mode == "hard" else (lambda s: entropy_top3(gi[s], w))
        o = H(sel); span = pos[-1] - pos[0] + 1; base = sel.min(); rel = pos[sel] - pos[base]; nul = []
        for _ in range(nperm):
            off = rng.integers(1, span); newp = pos[0] + (pos[base] - pos[0] + off + rel) % span; s2 = np.searchsorted(pos, newp).clip(0, n - 1)
            s2 = np.where(np.abs(pos[s2] - newp) <= np.abs(pos[(s2 - 1).clip(0)] - newp), s2, (s2 - 1).clip(0)); nul.append(H(s2))
        nul = np.array(nul); ob.append(o); nl.append(nul)
        dom = np.bincount(lab[sel], weights=w, minlength=ncat).argmax()
        rows.append(dict(locus=lc, n_var=len(sel), neff=neff, n_cs=pp.cs.nunique(), H=o, H_norm=o / np.log(ncat), eff_ctx=np.exp(o), H_null=nul.mean(), z=(o - nul.mean()) / (nul.std() + 1e-12),
                         p=(1 + (nul <= o).sum()) / (len(nul) + 1), dom=dom, dom_w=np.bincount(lab[sel], weights=w, minlength=ncat).max()))
    R = pd.DataFrame(rows); ob, nl = np.array(ob), np.array(nl); mo, mn = ob.mean(), nl.mean(0); p = (1 + (mn <= mo).sum()) / (len(mn) + 1)
    return R, mo, mn, p
def name(k, ncat): return CL[k] if ncat == NC else (("발현·" if k < NC else "스플라이싱·") + CL[k % NC])
P("\n=== G1 (주): fine-mapped 변이(P0 CS, PIP 가중)의 세포 맥락 엔트로피 — 낮을수록 수렴 ===")
R, mo, mn, p = G(save=True)
P(f"  좌위 {len(R)}개 (유효 변이 수 ≥ 2) | 평균 H {mo:.3f} (정규화 {mo / np.log(NC):.3f}, 유효 맥락 수 {np.exp(R.H).mean():.2f}) vs 귀무 {mn.mean():.3f} (95 % 하한 {np.quantile(mn, .05):.3f}) | 단측 p {p:.4f} → {'수렴 지지' if p < .05 else '불지지'}")
R["q"] = multipletests(R.p, method="fdr_bh")[1]; nn = int((R.p < .05).sum())
P(f"  수렴 좌위: 명목 {nn}/{len(R)} (귀무 기대 {.05 * len(R):.1f}, 이항 p {stats.binomtest(nn, len(R), .05, alternative='greater').pvalue:.3g}) | BH-FDR < 0.1: {int((R.q < .1).sum())}")
for r in R.sort_values("p").head(12).itertuples():
    P(f"    {r.locus:20s} H {r.H:.2f} (귀무 {r.H_null:.2f}, z {r.z:+.1f}, p {r.p:.3f}, q {r.q:.2f}) | 변이 {r.n_var} (유효 {r.neff:.1f}), CS {r.n_cs} | 주 맥락 {name(r.dom, NC)} ({r.dom_w:.0%})")
P("\n=== G2 (부): 세포 × 기전 엔트로피 (24범주) ===")
R2, mo2, mn2, p2 = G("lab_cm", 2 * NC)
R2["q"] = multipletests(R2.p, method="fdr_bh")[1]
P(f"  평균 H {mo2:.3f} vs 귀무 {mn2.mean():.3f} | 단측 p {p2:.4f} | 수렴 좌위 명목 {int((R2.p < .05).sum())}/{len(R2)}, BH < 0.1: {int((R2.q < .1).sum())}")
for r in R2.sort_values("p").head(10).itertuples(): P(f"    {r.locus:20s} H {r.H:.2f} (귀무 {r.H_null:.2f}, p {r.p:.3f}) | 주 {name(r.dom, 2 * NC)} ({r.dom_w:.0%})")
P("\n=== G3 (부): CS 간 대표 변이 세포 맥락 일치 (CS ≥ 2 좌위) ===")
lead = PIP.sort_values("pip").groupby(["locus", "cs"]).tail(1).merge(D[["atlas_variant", "lab"]], on="atlas_variant")
multi = lead.groupby("locus").filter(lambda x: len(x) >= 2)
def agree(df):
    a = []
    for _, g_ in df.groupby("locus"):
        l = g_.lab.to_numpy(); iu = np.triu_indices(len(l), 1); a += list(l[iu[0]] == l[iu[1]])
    return np.mean(a), len(a)
ob3, npair = agree(multi); nl3 = []
for _ in range(1000): m2 = multi.copy(); m2["lab"] = rng.permutation(multi.lab.to_numpy()); nl3.append(agree(m2)[0])
P(f"  좌위 {multi.locus.nunique()}개, CS 쌍 {npair}개: 일치율 {ob3:.3f} vs 귀무 {np.mean(nl3):.3f} | 단측 p {(1 + (np.array(nl3) >= ob3).sum()) / 1001:.3f}")
P("\n=== 민감도 (G1) ===")
for lab_, kw in [("상위 3개 부류 1/3 가중", dict(mode="top3", nperm=500)), ("보정 없는 원 최대 트랙 부류", dict(labcol="lab_raw", nperm=500)), ("PIP ≥ 0.1 변이만", dict(pipmin=.1, nperm=500))]:
    Rs, m_, n_, p_ = G(**kw); P(f"  {lab_:22s} 좌위 {len(Rs)} | 평균 H {m_:.3f} vs 귀무 {n_.mean():.3f} | 단측 p {p_:.4f} | 명목 수렴 좌위 {int((Rs.p < .05).sum())}")
R["dom_name"] = R.dom.map(lambda k: name(k, NC)); R.merge(R2[["locus", "H", "p", "dom"]].rename(columns={"H": "H_cm", "p": "p_cm", "dom": "dom_cm"}), on="locus", how="left").assign(dom_cm_name=lambda d: d.dom_cm.map(lambda k: name(int(k), 2 * NC) if k == k else "")).to_csv(RES/"e08_locus_entropy.csv", index=False)
(RES/"e08_entropy.txt").write_text("\n".join(log), encoding="utf-8")
