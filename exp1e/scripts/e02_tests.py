"""실험 1e e02: E1(연관 ↔ AVI, 좌위 간 일정성) · E2(상위 변이 기전 종류 수렴, T4 와 같은 귀무·LD 층화) · E3(주된 종류) · 양성 대조(PILRA rs1859788) · 민감도.
출력: results/e02_tests.txt, e02_locus.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"; W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e")
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2", "maf", "ldscore", "tss_dist"]
D = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
A = pd.concat([pd.read_parquet(f) for f in sorted(X.glob("part_*.parquet"))], ignore_index=True).drop_duplicates("atlas_variant")
D = D.merge(A, on="atlas_variant", how="inner").sort_values(["locus", "pos"]).reset_index(drop=True); D["log_tss"] = np.log10(D.tss_dist + 1)
CLS = {"발현": ["RNA_SEQ", "CAGE", "PROCAP", "POLYADENYLATION"], "조절": ["DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS"], "스플라이싱": ["MERGED_SPLICING"],
       "단백질": ["ALPHAMISSENSE", "PROTEIN_TERMINATION", "START_LOST", "STOP_LOST"], "보존": ["CACTUS_241_WAY", "PHASTCONS_470_WAY"]}
col = lambda f: f"fi_{f}" if f in ("MERGED_SPLICING", "ALPHAMISSENSE", "PROTEIN_TERMINATION", "START_LOST", "STOP_LOST", "CACTUS_241_WAY", "PHASTCONS_470_WAY") else f"fi_MAX_ABS_{f}"
CN = list(CLS)
def class_scores(signed=False):
    f = (lambda v: v) if signed else np.abs
    return np.column_stack([sum(f(D[col(x)].astype(float).fillna(0)) for x in CLS[c]) for c in CN])
Kab = class_scores(); Ksg = class_scores(signed=True)
for j, c in enumerate(CN): D[f"K_{c}"] = Kab[:, j]
tot = Kab.sum(1); SH = np.divide(Kab, tot[:, None], out=np.zeros_like(Kab), where=tot[:, None] > 0); DOM = np.where(tot > 0, Kab.argmax(1), -1)
P(f"SNV {len(D):,} (좌위 {D.locus.nunique()}); AVI 있음 {int(D.avi.notna().sum()):,}")
P("전체 SNV 주 종류 분포: " + ", ".join(f"{CN[k] if k >= 0 else '없음'} {np.mean(DOM == k):.1%}" for k in [-1] + list(range(len(CN)))))
P("평균 종류 비율: " + ", ".join(f"{c} {SH[:, j].mean():.3f}" for j, c in enumerate(CN)))
rng = np.random.default_rng(20260926)
CONF = ["ldscore", "maf", "log_tss"]
def rres(v, Z):
    rv = stats.rankdata(v); Zr = np.column_stack([np.ones(len(v))] + [stats.rankdata(z) for z in Z.T]); return rv - Zr @ np.linalg.lstsq(Zr, rv, rcond=None)[0]
def locus_rho(score, loci=None, nperm=1000):
    obs, nul, names = [], [], []
    for lc, g in D.groupby("locus", sort=False):
        if loci is not None and lc not in loci: continue
        Z = g[CONF].to_numpy(float); y = rres(g.chi2.to_numpy(), Z); s = np.nan_to_num(score[g.index].astype(float)); n = len(s)
        o = np.corrcoef(rres(s, Z), y)[0, 1]; sh = rng.integers(max(1, n // 20), n - max(1, n // 20), nperm)
        obs.append(o); nul.append([np.corrcoef(rres(np.roll(s, k), Z), y)[0, 1] for k in sh]); names.append(lc)
    return np.array(obs), np.array(nul), names
def e1_report(tag, score, loci=None, het=False, nperm=1000):
    o, N, nm = locus_rho(score, loci, nperm); rb = o.mean(); nb = N.mean(0); p = (1 + (nb >= rb).sum()) / (len(nb) + 1)
    P(f"  {tag:30s} ρ̄ {rb:+.4f} (귀무 {nb.mean():+.4f}, 95 % {np.quantile(nb, .95):+.4f}) 단측 p {p:.4f} | 양의 ρ 좌위 {np.mean(o > 0):.0%} (귀무 {np.mean(N > 0):.0%})")
    if het:
        mu, sd = N.mean(1, keepdims=True), N.std(1, keepdims=True) + 1e-12; z = (o[:, None] - mu) / sd; zb = (N - mu) / sd
        H = ((z - z.mean()) ** 2).sum(); Hb = ((zb - zb.mean(0)) ** 2).sum(0); ph = 2 * min((1 + (Hb >= H).sum()) / (len(Hb) + 1), (1 + (Hb <= H).sum()) / (len(Hb) + 1))
        return o, N, nm, z[:, 0], H, Hb, ph, p
    return o, N, nm, None, None, None, None, p
# ---------- E1 ----------
P("\n=== E1a (주): 좌위 안 부분 Spearman ρ(AVI, χ² | LD 점수, MAF, TSS 거리), 75 좌위 평균; 원형 이동 1,000회 ===")
o, N, nm, z, H, Hb, ph, p1 = e1_report("AVI", D.avi.to_numpy(), het=True)
P(f"  → E1a {'통과' if p1 < .05 else '불통과'} (α 0.05 단측)")
P(f"\n=== E1b: 좌위 간 일정성 ===\n  이질성 H = {H:.1f} (귀무 중앙 {np.median(Hb):.1f}, 95 % {np.quantile(Hb, .95):.1f}), 양측 p {ph:.3f}")
g50 = np.array([n.startswith("L") for n in nm]); P(f"  1단계 50 좌위 ρ̄ {o[g50].mean():+.4f} (양의 비율 {np.mean(o[g50] > 0):.0%}) | 재현 25 좌위 ρ̄ {o[~g50].mean():+.4f} (양의 비율 {np.mean(o[~g50] > 0):.0%})")
v1b = "좌위 간 일정" if (ph >= .05 and np.mean(o > 0) > np.mean(N > 0)) else "좌위 간 이질"; P(f"  → E1b 판정: {v1b}")
Lr = pd.DataFrame({"locus": nm, "rho_avi": o, "z_avi": z}); zz = Lr.sort_values("z_avi")
P("  이탈 좌위(z 하위 5): " + ", ".join(f"{r.locus.split('_')[0]} {r.z_avi:+.1f}" for r in zz.head(5).itertuples()) + " | 상위 5: " + ", ".join(f"{r.locus.split('_')[0]} {r.z_avi:+.1f}" for r in zz.tail(5)[::-1].itertuples()))
P("\n=== E1c (기술): 종류 점수별 ρ̄ ===")
for j, c in enumerate(CN):
    oc, Nc, _, _, _, _, _, _ = e1_report(f"K_{c}", Kab[:, j]); Lr[f"rho_{c}"] = oc
# ---------- E2 / E3 ----------
LDC = {}
def ld_r2(lc, av):
    if lc not in LDC: z = np.load(W/"ld"/f"{lc}.npz", allow_pickle=True); LDC[lc] = (pd.Index(z["variant"].astype(str)), z["R"].astype(np.float32))
    v, R_ = LDC[lc]; ix = v.get_indexer(av); out = np.full((len(av), len(av)), np.nan); m = ix >= 0; out[np.ix_(m, m)] = R_[np.ix_(ix[m], ix[m])].astype(float) ** 2; return out
def e2(frac, nperm=1000, dom=DOM, sh=SH, loci=None, save=False):
    shc = sh - sh.mean(0); rows, ob, nls = [], [], []
    for lc, g in D.groupby("locus", sort=False):
        if loci is not None and lc not in loci: continue
        idx = g.index.to_numpy(); pos = g.pos.to_numpy(); n = len(idx); k = max(5, int(round(frac * n)))
        top = np.sort(np.argsort(-g.chi2.to_numpy())[:k]); iu = np.triu_indices(k, 1); r2 = ld_r2(lc, g.atlas_variant.to_numpy()[top])[iu]; lo, hi = r2 < .1, r2 >= .5
        def st(sel):
            ii = idx[sel]; d = dom[ii]; a, b = d[iu[0]], d[iu[1]]; has = (a >= 0) & (b >= 0); same = (a == b) & has
            V = shc[ii]; nv = np.linalg.norm(V, axis=1) + 1e-12; C = (V @ V.T) / np.outer(nv, nv); c = C[iu]
            f = lambda m: same[m & has].mean() if (m & has).any() else np.nan
            return np.array([f(np.ones_like(has)), c.mean(), f(lo), c[lo].mean() if lo.any() else np.nan, f(hi)] + list(sh[ii].mean(0) - sh[idx].mean(0)))
        o_ = st(top); span = pos[-1] - pos[0] + 1; rel = pos[top] - pos[top[0]]; nul = []
        for _ in range(nperm):
            off = rng.integers(1, span); newp = pos[0] + (pos[top[0]] - pos[0] + off + rel) % span; sel = np.searchsorted(pos, newp).clip(0, n - 1)
            sel = np.where(np.abs(pos[sel] - newp) <= np.abs(pos[(sel - 1).clip(0)] - newp), sel, (sel - 1).clip(0)); nul.append(st(sel))
        nul = np.array(nul); ob.append(o_); nls.append(nul)
        zc = (o_ - np.nanmean(nul, 0)) / (np.nanstd(nul, 0) + 1e-12)
        rows.append(dict(locus=lc, k=k, n_lowld=int(lo.sum()), agree=o_[0], agree_null=np.nanmean(nul[:, 0]), z_agree=zc[0], cos=o_[1], z_cos=zc[1], agree_lo=o_[2], z_agree_lo=zc[2],
                         top_class=CN[int(np.argmax(sh[idx[top]].mean(0)))], **{f"dshare_{c}": o_[5 + j] for j, c in enumerate(CN)}, **{f"z_dshare_{c}": zc[5 + j] for j, c in enumerate(CN)}))
    ob = np.array(ob); nls = np.array(nls); R = pd.DataFrame(rows); out = {}
    for lab, j in [("주 종류 일치율(전체)", 0), ("종류 비율 코사인(전체)", 1), ("주 종류 일치율(낮은 LD)", 2), ("주 종류 일치율(높은 LD)", 4)]:
        m = ~np.isnan(ob[:, j]); o_ = ob[m, j].mean(); Nn = np.nanmean(nls[m, :, j], 0); p = (1 + (Nn >= o_).sum()) / (len(Nn) + 1); out[lab] = p
        P(f"  {lab:24s} 관측 {o_:.4f} vs 귀무 {np.nanmean(Nn):.4f} | 단측 p {p:.4f} ({int(m.sum())} 좌위)")
    if save: return out, R, ob, nls
    return out, R
P("\n=== E2 (주): 좌위별 χ² 상위 1 % 의 기전 종류 수렴 (거리 보존 평행이동 1,000회) ===")
o2, R2, ob, nls = e2(.01, save=True)
v2 = "지지" if (o2["주 종류 일치율(전체)"] < .05 and o2["주 종류 일치율(낮은 LD)"] < .05) else ("태그·인접 효과" if o2["주 종류 일치율(전체)"] < .05 else "불지지")
P(f"  → E2 판정: {v2}")
P("\n=== E3 (기술): 상위 집합의 종류 비율 − 좌위 배경 (좌위 평균, 평행이동 귀무 대비) ===")
for j, c in enumerate(CN):
    o_ = ob[:, 5 + j].mean(); Nn = nls[:, :, 5 + j].mean(0); P(f"  {c:6s} Δ비율 {o_:+.4f} (귀무 {Nn.mean():+.4f}), 양측 p {2 * min((1 + (Nn >= o_).sum()), (1 + (Nn <= o_).sum())) / (len(Nn) + 1):.3f}")
P("  좌위별 상위 집합 주된 종류: " + ", ".join(f"{k} {v}" for k, v in R2.top_class.value_counts().items()))
V = R2[[f"dshare_{c}" for c in CN]].to_numpy(); nv = np.linalg.norm(V, axis=1) + 1e-12; Cm = (V @ V.T) / np.outer(nv, nv); iu = np.triu_indices(len(V), 1)
Vn = nls[:, :, 5:]; cn_ = []
for b in range(Vn.shape[1]):
    Vb = Vn[:, b, :]; nb = np.linalg.norm(Vb, axis=1) + 1e-12; cn_.append(((Vb @ Vb.T) / np.outer(nb, nb))[iu].mean())
P(f"  좌위 간 일관성(Δ비율 벡터 평균 코사인) {Cm[iu].mean():+.3f} (귀무 {np.mean(cn_):+.3f}), 단측 p {(1 + (np.array(cn_) >= Cm[iu].mean()).sum()) / (len(cn_) + 1):.3f}")
# ---------- 양성 대조 ----------
P("\n=== 양성 대조: PILRA rs1859788 (G78R) ===")
pc = D[D.variant_id == "rs1859788"]
if len(pc):
    i = pc.index[0]; P(f"  {pc.atlas_variant.iat[0]} 주 종류 {CN[DOM[i]] if DOM[i] >= 0 else '없음'} | 종류 비율 " + ", ".join(f"{c} {SH[i, j]:.2f}" for j, c in enumerate(CN)) + f" | AVI {pc.avi.iat[0]:.3f} (분위 {pc.avi_q.iat[0]:.3f}) | χ² {pc.chi2.iat[0]:.1f}")
    P(f"  → 분류 작동 여부: {'작동(단백질)' if DOM[i] == CN.index('단백질') else '경고: 단백질이 주 종류가 아님'}")
    r = R2[R2.locus.str.startswith("L009")]
    if len(r): P(f"  PILRA 좌위 상위 집합 단백질 Δ비율 {r['dshare_단백질'].iat[0]:+.3f} (z {r['z_dshare_단백질'].iat[0]:+.1f}), 상위 집합 주된 종류 {r.top_class.iat[0]}")
else: P("  rs1859788 없음")
# ---------- 민감도 ----------
P("\n=== 민감도 (기술) ===")
for f in [.005, .02]: P(f"  [E2 상위 {f:.1%}]"); e2(f, nperm=500)
keep = [l for l in D.locus.unique() if not l.startswith(("L006", "L028"))]
P("  [L006·L028 제외]"); e1_report("AVI", D.avi.to_numpy(), loci=set(keep), nperm=500); e2(.01, nperm=500, loci=set(keep))
tsg = np.abs(Ksg).sum(1); SHs = np.divide(np.abs(Ksg), tsg[:, None], out=np.zeros_like(Ksg), where=tsg[:, None] > 0); DOMs = np.where(tsg > 0, np.abs(Ksg).argmax(1), -1)
P("  [부호 있는 기여 합으로 종류 정의]"); e2(.01, nperm=500, dom=DOMs, sh=SHs)
P("  [AVI 분위수]"); e1_report("AVI 분위수", D.avi_q.to_numpy(), nperm=500)
Lr.merge(R2, on="locus").to_csv(RES/"e02_locus.csv", index=False); (RES/"e02_tests.txt").write_text("\n".join(log), encoding="utf-8")
