"""실험 1e-5 e11: 선도 SNP vs 연관 SNP 의 위험 방향 프로필 일치 (PROTOCOL_exp1e5_2026-09-27).
D0 세포별 방향 분할 빈도 · D1 CS 구성원(주) · D2 상위 1 % · D3 LD 층 · D4 CS 간 · D5 SNP 분류 · 민감도.
출력: results/e11_direction.txt, e11_snp_class.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"; W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e/signed")
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
V = pd.read_parquet(X/"variants.parquet")
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2", "ref", "alt", "effect_allele", "beta"]
A = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]]).drop_duplicates("atlas_variant")
V = V.merge(A.drop(columns=["locus"]), on="atlas_variant")
V["s"] = np.where(V.effect_allele == V.alt, np.sign(V.beta), np.where(V.effect_allele == V.ref, -np.sign(V.beta), np.nan))
P(f"변이 {len(V):,}; 위험 방향 결정 불가(대립유전자 불일치) {int(V.s.isna().sum())} 제외"); V = V[V.s.notna() & (V.s != 0)]
Z = [np.load(f, allow_pickle=True) for f in sorted(X.glob("part_*.npz"))]
Q = np.vstack([z["Q"] for z in Z]).astype(np.float32); vv = np.concatenate([z["variant"] for z in Z]); gg = np.concatenate([z["gene"] for z in Z])
IDX = pd.Series(np.arange(len(vv)), index=pd.MultiIndex.from_arrays([vv, gg])); IDX = IDX[~IDX.index.duplicated()]
SV = V.set_index("atlas_variant").s; LV0 = set(IDX.index.get_level_values(0))
def prof(v, g, orient=True):
    i = IDX.get((v, g));
    return None if i is None else Q[i] * (SV[v] if orient else 1.0)
def top_gene(v):
    rows = IDX.xs(v, level=0) if v in LV0 else None
    if rows is None or len(rows) == 0: return None
    return rows.index[np.argmax(np.abs(Q[rows.to_numpy()]).max(1))]
rho = lambda a, b: stats.spearmanr(a, b).statistic
def sc(a, b):
    m = (np.abs(a) >= .5) & (np.abs(b) >= .5); return np.mean(np.sign(a[m]) == np.sign(b[m])) if m.sum() >= 5 else np.nan
# ---------- D0 ----------
P("\n=== D0 (기술): 세포에 따라 방향이 갈리는 SNP 는 흔한가 (자기 최대 유전자, |q| ≥ 0.5 트랙 중 소수 부호 비율) ===")
mins, ntr = [], []
for v in V.atlas_variant:
    g = top_gene(v)
    if g is None: continue
    q = Q[IDX[(v, g)]]; m = np.abs(q) >= .5
    if m.sum() >= 5: pos = (q[m] > 0).mean(); mins.append(min(pos, 1 - pos)); ntr.append(m.sum())
mins = np.array(mins); P(f"  SNV {len(mins):,} (|q| ≥ 0.5 트랙 5개 이상) | 소수 부호 비율 중앙 {np.median(mins):.3f}, 평균 {mins.mean():.3f} | 0 (전부 같은 부호) {np.mean(mins == 0):.1%} | ≥ 10 % {np.mean(mins >= .1):.1%} | ≥ 25 % {np.mean(mins >= .25):.1%}")
# ---------- 선도 · 표적 유전자 · 거리 맞춤 후보 ----------
LDC = {}
def r2(lc, a, b):
    if lc not in LDC: z = np.load(W/"ld"/f"{lc}.npz", allow_pickle=True); LDC[lc] = (pd.Index(z["variant"].astype(str)), z["R"].astype(np.float32))
    ix, R_ = LDC[lc]; i, j = ix.get_indexer([a, b]); return float(R_[i, j] ** 2) if i >= 0 and j >= 0 else np.nan
PIP = pd.concat([pd.read_csv(f).assign(locus=f.name.replace("_pip.csv", "")) for d in [W/"s2"/"out_P0", W/"s2_rep25"/"out_P0"] for f in d.glob("*_pip.csv")]); PIP = PIP[PIP.cs.notna()]
rng = np.random.default_rng(20261002); rows = []; NULLPOOL = {}
for lc, g in V.groupby("locus"):
    L = g[g.role.str.contains("lead")]
    if L.empty: continue
    lv = L.atlas_variant.iat[0]; lp = L.pos.iat[0]; tg = top_gene(lv)
    if tg is None: continue
    pl = prof(lv, tg); pl0 = prof(lv, tg, orient=False)
    rnd = g[g.role == "random;"]; rnd = rnd[[(v, tg) in IDX.index for v in rnd.atlas_variant]]
    rp = {v: prof(v, tg) for v in rnd.atlas_variant}; rrho = {v: rho(pl, p_) for v, p_ in rp.items()}; rd = np.abs(rnd.pos.to_numpy() - lp); rv = rnd.atlas_variant.to_numpy()
    NULLPOOL[lc] = (rv, rd, np.array([rrho[v] for v in rv]), np.array([sc(pl, rp[v]) for v in rv]))
    for r in g[~g.role.str.contains("lead") & (g.role != "random;")].itertuples():
        p_ = prof(r.atlas_variant, tg)
        if p_ is None: continue
        d = abs(r.pos - lp); ok = (rd >= .5 * d) & (rd <= 2 * d)
        if ok.sum() < 3: ok = np.argsort(np.abs(rd - d))[:max(3, min(10, len(rd)))]; ok = np.isin(np.arange(len(rd)), ok)
        nr = NULLPOOL[lc][2][ok]; o = rho(pl, p_)
        rows.append(dict(locus=lc, variant=r.atlas_variant, rsid=r.variant_id, role=r.role, gene=tg, dist=d, r2=r2(lc, lv, r.atlas_variant), rho=o, sc=sc(pl, p_),
                         rho_noorient=rho(pl0, prof(r.atlas_variant, tg, orient=False)), pct=np.mean(nr < o) + .5 * np.mean(nr == o), cand=np.where(ok)[0]))
R = pd.DataFrame(rows); P(f"\n분석 좌위 {R.locus.nunique()}; 비교 SNP {len(R):,} (CS {int(R.role.str.contains('cs').sum())}, 상위 1 % 비 CS {int((~R.role.str.contains('cs')).sum())})")
def test(sub, col="rho", ncol=2, nperm=1000, tag=""):
    loci = sub.groupby("locus"); obs = np.nanmean([x[col].mean() for _, x in loci]); nl = []
    for _ in range(nperm):
        m = []
        for lc, x in loci:
            pool = NULLPOOL[lc][ncol]; m.append(np.nanmean([pool[rng.choice(c)] for c in x.cand]))
        nl.append(np.nanmean(m))
    nl = np.array(nl); p = (1 + (nl >= obs).sum()) / (len(nl) + 1)
    P(f"  {tag:34s} 좌위 {sub.locus.nunique():2d}, SNP {len(sub):5d} | 평균 {obs:+.4f} vs 귀무 {nl.mean():+.4f} (95 % {np.quantile(nl, .95):+.4f}) | 단측 p {p:.4f}"); return p
P("\n=== D1 (주): CS 구성원 vs 선도 — 위험 방향 프로필 Spearman (거리 맞춤 귀무 1,000회) ===")
cs_ = R[R.role.str.contains("cs")]; p1 = test(cs_, tag="CS 구성원"); P(f"  → D1 {'지지' if p1 < .05 else '불지지'}")
P("\n=== D2: χ² 상위 1 % (CS 밖) vs 선도 ===")
test(R[~R.role.str.contains("cs")], tag="상위 1 % 비 CS")
P("\n=== D3: LD 층 (CS + 상위 1 %) ===")
for lab, m in [("r² ≥ 0.5 (편승 가능)", R.r2 >= .5), ("0.1 ≤ r² < 0.5", (R.r2 >= .1) & (R.r2 < .5)), ("r² < 0.1 (독립 가능)", R.r2 < .1)]:
    if m.sum(): test(R[m], tag=lab)
P("\n=== D4 (기술): 부 CS 대표 변이 vs 선도 (CS ≥ 2 좌위) ===")
leadcs = {}
for lc, g in PIP.groupby("locus"):
    lv = V[(V.locus == lc) & V.role.str.contains("lead")].atlas_variant
    for c, gc in g.groupby("cs"):
        if len(lv) and lv.iat[0] in set(gc.atlas_variant): continue
        rep = gc.sort_values("pip").atlas_variant.iat[-1]; hit = R[(R.locus == lc) & (R.variant == rep)]
        if len(hit): leadcs[(lc, c)] = (hit.rho.iat[0], hit.pct.iat[0], hit.r2.iat[0], hit.rsid.iat[0])
if leadcs:
    arr = np.array([v[1] for v in leadcs.values()]); P(f"  부 CS 대표 {len(leadcs)}개: ρ 평균 {np.mean([v[0] for v in leadcs.values()]):+.3f} | 귀무 대비 백분위 평균 {arr.mean():.2f} (귀무 기대 0.50) | ≥ 0.95 {int((arr >= .95).sum())}, ≤ 0.05 {int((arr <= .05).sum())}")
    for (lc, c), (r_, pc, rr, rs) in sorted(leadcs.items(), key=lambda x: -x[1][1])[:8]: P(f"    {lc:20s} CS{int(c)} {rs} ρ {r_:+.2f} 백분위 {pc:.2f} r² {rr:.2f}")
P("\n=== D5 (기술): 연관 SNP 분류 (거리 맞춤 무작위 대비 백분위; 귀무 기대 각 5 %) ===")
R["class"] = np.select([R.pct >= .95, R.pct <= .05], ["선도와 같은 방향", "반대 방향"], "무관"); R["ld"] = np.select([R.r2 >= .5, R.r2 < .1], ["높은 LD", "낮은 LD"], "중간")
tab = pd.crosstab(R.ld, R["class"], normalize="index").round(3); P("  " + tab.to_string().replace("\n", "\n  "))
P(f"  전체: 같은 방향 {np.mean(R['class'] == '선도와 같은 방향'):.1%}, 반대 {np.mean(R['class'] == '반대 방향'):.1%} (각 귀무 기대 5 %); 이항 p(같은 방향 > 5 %) {stats.binomtest(int((R['class'] == '선도와 같은 방향').sum()), len(R), .05, alternative='greater').pvalue:.2e}")
lt = R.groupby("locus").agg(n=("rho", "size"), same=("class", lambda s: (s == "선도와 같은 방향").sum()), opp=("class", lambda s: (s == "반대 방향").sum()), gene=("gene", "first"), same_lowld=("class", lambda s: 0))
lt["same_lowld"] = R[(R["class"] == "선도와 같은 방향") & (R.r2 < .1)].groupby("locus").size().reindex(lt.index).fillna(0).astype(int)
P("  같은 방향 SNP 가 많은 좌위 (상위 10):")
for lc, r in lt.sort_values("same", ascending=False).head(10).iterrows(): P(f"    {lc:20s} 표적 {r.gene:12s} 같은 방향 {r.same}/{r.n} (낮은 LD {r.same_lowld}), 반대 {r.opp}")
P("\n=== 민감도 (D1) ===")
test(cs_, col="sc", ncol=3, tag="보조 지표 SC (부호 일치율)")
def test_noorient(sub, nperm=1000):
    obs = np.nanmean([x.rho_noorient.mean() for _, x in sub.groupby("locus")]); P(f"  {'위험 방향 통일 없음 (ALT 기준) 관측':34s} 평균 {obs:+.4f} (통일 시 {np.nanmean([x.rho.mean() for _, x in sub.groupby('locus')]):+.4f}) — 귀무는 통일 기준이라 관측값만 기술")
test_noorient(cs_)
def shared_gene_rho(sub, pool_role="random;"):
    def val(lc, v):
        lv = V[(V.locus == lc) & V.role.str.contains("lead")].atlas_variant.iat[0]; out = []
        for g in (IDX.xs(lv, level=0).index if lv in LV0 else []):
            a, b = prof(lv, g), prof(v, g)
            if b is not None and np.abs(a).max() >= .9 and np.abs(b).max() >= .9: out.append(rho(a, b))
        return np.mean(out) if out else np.nan
    o = np.nanmean([val(r.locus, r.variant) for r in sub.itertuples()])
    rs = V[V.role == pool_role].groupby("locus").head(10); n_ = np.nanmean([val(r.locus, r.atlas_variant) for r in rs.itertuples() if r.locus in set(sub.locus)])
    P(f"  {'공통 표적 유전자(둘 다 |q| ≥ 0.9) 평균 ρ':34s} CS 구성원 {o:+.4f} vs 무작위(좌위당 10개, 거리 무관) {n_:+.4f} — 기술")
shared_gene_rho(cs_)
R.drop(columns=["cand"]).to_csv(RES/"e11_snp_class.csv", index=False); (RES/"e11_direction.txt").write_text("\n".join(log), encoding="utf-8")
