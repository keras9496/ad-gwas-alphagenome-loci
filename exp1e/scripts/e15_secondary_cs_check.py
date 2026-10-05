"""실험 1e e15: 원고 C3(R67) 대응 — 방향 프로필 D4(부 CS 대표 변이 vs 선도)의 CS별 표, LD 불일치 의심 좌위 제외 민감도 (2026-10-05, 사후).
 e11의 자료 준비·비교 행 계산(R, NULLPOOL)을 소스에서 그대로 실행해 같은 값을 다시 만든다(무작위 요소 없음).
 추가로 각 대표 변이의 백분위가 몇 개의 거리 맞춤 무작위 SNV(m)와 비교한 값인지 센다.
   백분위 = mean(귀무 < 관측) + 0.5·mean(귀무 = 관측). m < 20이면 "≥ 0.95"는 m개 모두보다 큰 경우뿐이다. 그래서 귀무 확률은 0.05가 아니라 (m − ⌈0.95m⌉ + 1)/(m + 1)이다.
   → 이항(p = 0.05) 외에, CS마다 정확한 귀무 확률을 쓰는 포아송-이항 꼬리 확률도 함께 보고한다.
 D1(주, CS 구성원 vs 선도)도 L006·L028을 빼고 e11과 같은 방법(거리 맞춤 귀무 1,000회, 시드 20261002)으로 다시 계산한다.
출력: results/e15_secondary_cs/posthoc_secondary_cs.csv, e15_secondary_cs.txt
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
EXP = Path(__file__).resolve().parents[1]; OUT = EXP/"results"/"e15_secondary_cs"; OUT.mkdir(exist_ok=True)
src = (EXP/"scripts"/"e11_direction.py").read_text(encoding="utf-8")
ns = {"__file__": str(EXP/"scripts"/"e11_direction.py")}
exec(src[:src.index('P("\\n=== D1')], ns)          # e11 1–74행: 자료, 프로필, R, NULLPOOL, test()
R, PIP, V, NULLPOOL, test, W = ns["R"], ns["PIP"], ns["V"], ns["NULLPOOL"], ns["test"], ns["W"]
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
MIS = {"L006_chr11_60254k", "L028_chr16_30010k"}
CSF = pd.concat([pd.read_csv(f) for d in [W/"s2"/"out_P0", W/"s2_rep25"/"out_P0"] for f in d.glob("*_cs.csv")]).dropna(subset=["cs"])
rows = []
for lc, g in PIP.groupby("locus"):
    lv = V[(V.locus == lc) & V.role.str.contains("lead")].atlas_variant
    for c, gc in g.groupby("cs"):
        if len(lv) and lv.iat[0] in set(gc.atlas_variant): continue
        rep = gc.sort_values("pip").atlas_variant.iat[-1]; hit = R[(R.locus == lc) & (R.variant == rep)]
        if not len(hit): continue
        h = hit.iloc[0]; m = len(h.cand); k95 = int(np.ceil(.95 * m)); p_null = (m - k95 + 1) / (m + 1)
        info = CSF[(CSF.locus == lc) & (CSF.cs == c)]
        rows.append(dict(locus=lc, ld_mismatch_locus="yes" if lc in MIS else "no", cs=int(c), cs_size=int(info["size"].iat[0]) if len(info) else len(gc),
                         cs_purity=float(info.purity.iat[0]) if len(info) else np.nan, rep_variant=rep, rep_rsid=h.rsid, rep_pip=float(gc.pip.max()),
                         r2_with_lead=h.r2, target_gene=h.gene, rho_dir=h.rho, percentile=h.pct, n_null_snvs=m, null_prob_pct_ge_0_95=p_null))
T = pd.DataFrame(rows).sort_values(["percentile", "rho_dir"], ascending=False)
T.to_csv(OUT/"posthoc_secondary_cs.csv", index=False)
assert len(T) == 35 and int((T.percentile >= .95).sum()) == 6, (len(T), int((T.percentile >= .95).sum()))
P(f"부 CS 대표 {len(T)}개 (e11 D4와 같음: ≥ 0.95 {int((T.percentile >= .95).sum())}개)")
P("좌위별 부 CS 수: " + ", ".join(f"{k} {v}" for k, v in T.locus.value_counts().items()))
P("\n=== L006·L028의 부 CS ===")
for r in T[T.ld_mismatch_locus == "yes"].itertuples():
    P(f"  {r.locus} CS{r.cs}: 크기 {r.cs_size}, 순도 {r.cs_purity:.2f}, 대표 {r.rep_rsid} (PIP {r.rep_pip:.3f}), lead r² {r.r2_with_lead:.2f}, 유전자 {r.target_gene}, ρ {r.rho_dir:+.2f}, 백분위 {r.percentile:.2f} (귀무 SNV {r.n_null_snvs})")
def pois_binom_tail(ps, k):   # P(X ≥ k), X = 독립 베르누이(ps)의 합
    dist = np.zeros(len(ps) + 1); dist[0] = 1
    for p in ps: dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p; dist[0] *= (1 - p)
    return dist[k:].sum()
P("\n=== 95번째 백분위 초과 수 검정 ===")
for lab, sub in [("전체", T), ("L006·L028 제외", T[T.ld_mismatch_locus == "no"])]:
    n = len(sub); k = int((sub.percentile >= .95).sum())
    pb = stats.binomtest(k, n, .05, alternative="greater").pvalue; ps = sub.null_prob_pct_ge_0_95.to_numpy(); pp = pois_binom_tail(ps, k)
    P(f"  {lab}: n {n}, k {k}, 기대 0.05n {0.05 * n:.2f} → 이항 단측 P {pb:.4f} | 정확한 귀무 확률 합 {ps.sum():.2f} → 포아송-이항 단측 P {pp:.4f}")
P(f"  귀무 SNV 수 m: 중앙 {int(T.n_null_snvs.median())}, 범위 {int(T.n_null_snvs.min())}–{int(T.n_null_snvs.max())} | m < 20인 CS {int((T.n_null_snvs < 20).sum())}/{len(T)}")
P("\n=== D1 (주) 민감도: L006·L028 제외 (e11과 같은 방법, 거리 맞춤 귀무 1,000회) ===")
ns["rng"] = np.random.default_rng(20261002)
cs_ = R[R.role.str.contains("cs")]
test(cs_, tag="CS 구성원 (전체, 재계산)")
ns["rng"] = np.random.default_rng(20261002)
test(cs_[~cs_.locus.isin(MIS)], tag="CS 구성원 (L006·L028 제외)")
log += ns["log"][-2:]
(OUT/"e15_secondary_cs.txt").write_text("\n".join(log), encoding="utf-8")
