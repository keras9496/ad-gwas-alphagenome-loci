# [public version] GTEx portal query replaced by a read of a pre-downloaded eQTL tissue table; everything else as run.
"""실험 1e-3 e07: F1(연관 상위 SNV 의 최대 변화 세포 수렴) · F2(좌위별 세포 × 기전 정의 가능성) · F3(스플라이싱 기술) · GTEx 조직 대응 (PROTOCOL_exp1e3).
출력: results/e07_celltype.txt, e07_locus_celltype.csv
"""
import warnings; warnings.filterwarnings("ignore")
import re, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
EXP = Path(__file__).resolve().parents[1]; RES = EXP/"results"; W = Path("D:/AD_GWAS_Atlas_data/work"); X = Path("D:/AD_GWAS_Atlas_data/exp1e"); C = X/"ct"
log = []
def P(*x): s = " ".join(str(i) for i in x); print(s, flush=True); log.append(s)
# ---------- 세포 부류 (결과 보기 전 고정) ----------
CLASS_RULES = [
    ("신경세포", r"^(Purkinje cell|glutamatergic neuron|motor neuron|neural cell|neural progenitor cell|neuronal stem cell|neural crest cell|neurosphere|BE2C|SK-N-SH|SK-N-DZ|PFSK-1|Daoy)$"),
    ("교세포", r"^(astrocyte|A172|M059J|U-87 MG|H4)$"),
    ("골수계", r"^(CD14-positive monocyte|common myeloid progenitor, CD34-positive|hematopoietic multipotent progenitor cell|BLaER1|K562)$"),
    ("림프계", r"(B cell|T cell|T-cell|T-helper|natural killer|killer cell|^GM1289[12]$|^GM12878$|^Jurkat|^Karpas-422$|^OCI-LY7$|^lymphoblast$|^thymus$|Peyer's patch)"),
    ("혈액·면역 혼합", r"^(peripheral blood mononuclear cell|mononuclear cell|venous blood|spleen)$"),
    ("간", r"^(hepatocyte|liver|left lobe of liver|right lobe of liver|HepG2)$"),
    ("섬유아·중간엽", r"(fibroblast|^BJ$|^IMR-90$|^AG04450$|^HFFc6$|^GM23248$|mesenchymal stem cell|chondrocyte|osteoblast|osteocyte|^MG63$|^SJSA1$|^HT1080$|^A673$|dermal papilla|mesothelial|mesangial)"),
    ("대사·내분비", r"(adipose|fat pad|preadipocyte|pancrea|type B pancreatic cell|^Panc1$|adrenal gland|thyroid gland|pituitary gland)"),
    ("혈관", r"(endothelial|aorta|artery|arterial|vena cava|pericyte|aortic smooth muscle|smooth muscle cell of the (coronary|pulmonary|umbilical) artery)"),
    ("근육·심장", r"(heart|cardiac|myocardium|atrium|ventricle|septum|myocyte|myotube|myoblast|satellite cell|muscle|LHCN-M2|SJCRH30|myometrial|smooth muscle cell|gastrocnemius|muscularis)"),
    ("뇌 조직", r"^(Ammon's horn|amygdala|anterior cingulate cortex|brain|caudate nucleus|cerebellar hemisphere|cerebellum|diencephalon|dorsolateral prefrontal cortex|frontal cortex|hypothalamus|nucleus accumbens|occipital lobe|parietal lobe|putamen|substantia nigra|temporal lobe|spinal cord|C1 segment of cervical spinal cord)$"),
]
def cls(name):
    for c, rx in CLASS_RULES:
        if re.search(rx, name): return c
    return "상피·기타 장기"
TR = pd.read_csv(C/"tracks_RNA_SEQ.csv"); TR["cls"] = TR.biosample_name.astype(str).map(cls); TS = pd.read_csv(C/"tracks_SPLICE_JUNCTIONS.csv"); TS["cls"] = TS.biosample_name.astype(str).map(cls)
CL = sorted(TR.cls.unique()); P("RNA 트랙 부류 구성: " + ", ".join(f"{c} {n}" for c, n in TR.cls.value_counts().items()))
# ---------- 자료 ----------
K = ["atlas_variant", "variant_id", "locus", "pos", "chi2"]
D = pd.concat([pd.read_parquet(W/"locus_snps_annot.parquet")[K], pd.read_parquet(W/"rep25"/"snps_annot.parquet")[K]], ignore_index=True)
Z = [np.load(f, allow_pickle=True) for f in sorted(C.glob("rna_*.npz"))]; Qv = np.vstack([z["Q"] for z in Z]).astype(np.float32); vid = np.concatenate([z["variant"] for z in Z])
M = pd.concat([pd.read_parquet(f) for f in sorted(C.glob("rna_meta_*.parquet"))]).drop_duplicates("atlas_variant").set_index("atlas_variant")
D = D[D.atlas_variant.isin(M.index[M.tr.notna()])].sort_values(["locus", "pos"]).reset_index(drop=True)
D["tr"] = M.loc[D.atlas_variant, "tr"].astype(int).to_numpy(); D["gene"] = M.loc[D.atlas_variant, "gene"].to_numpy(); D["qmax"] = M.loc[D.atlas_variant, "q"].to_numpy()
D["bios"] = TR.biosample_name.to_numpy()[D.tr]; D["cls"] = TR.cls.to_numpy()[D.tr]; D["ci"] = D.cls.map({c: i for i, c in enumerate(CL)})
rowpos = pd.Series(np.arange(len(vid)), index=vid); RK = stats.rankdata(np.nan_to_num(Qv[rowpos.loc[D.atlas_variant].to_numpy()]), axis=1).astype(np.float32)
RK = (RK - RK.mean(1, keepdims=True)) / (RK.std(1, keepdims=True) + 1e-9)
P(f"SNV {len(D):,} (좌위 {D.locus.nunique()}); 최대 변화 세포 부류 분포: " + ", ".join(f"{c} {v:.1%}" for c, v in D.cls.value_counts(normalize=True).items()))
LDC = {}
def ld_r2(lc, av):
    if lc not in LDC: z = np.load(W/"ld"/f"{lc}.npz", allow_pickle=True); LDC[lc] = (pd.Index(z["variant"].astype(str)), z["R"].astype(np.float32))
    v, R_ = LDC[lc]; ix = v.get_indexer(av); out = np.full((len(av), len(av)), np.nan); m = ix >= 0; out[np.ix_(m, m)] = R_[np.ix_(ix[m], ix[m])].astype(float) ** 2; return out
rng = np.random.default_rng(20260928)
# ---------- F1 / F2 ----------
def run(frac=.01, nperm=1000, save=False):
    ob, nls, rows = [], [], []
    for lc, g in D.groupby("locus", sort=False):
        idx = g.index.to_numpy(); pos = g.pos.to_numpy(); n = len(idx); k = max(5, int(round(frac * n)))
        top = np.sort(np.argsort(-g.chi2.to_numpy())[:k]); iu = np.triu_indices(k, 1); r2 = ld_r2(lc, g.atlas_variant.to_numpy()[top]); lo, hi = r2[iu] < .1, r2[iu] >= .5
        ci, bs = g.ci.to_numpy(), g.bios.to_numpy(); dom = np.bincount(ci[top], minlength=len(CL)).argmax()
        def st(sel):
            a, b = ci[sel][iu[0]], ci[sel][iu[1]]; same = a == b; sb = bs[sel][iu[0]] == bs[sel][iu[1]]; R_ = RK[idx[sel]]; cc = (R_ @ R_.T / R_.shape[1])[iu]
            f = lambda v, m: v[m].mean() if m.any() else np.nan
            return np.array([same.mean(), f(same, lo), f(same, hi), sb.mean(), cc.mean(), f(cc, lo), np.mean(ci[sel] == dom)])
        o = st(top); span = pos[-1] - pos[0] + 1; rel = pos[top] - pos[top[0]]; nul = []
        for _ in range(nperm):
            off = rng.integers(1, span); newp = pos[0] + (pos[top[0]] - pos[0] + off + rel) % span; sel = np.searchsorted(pos, newp).clip(0, n - 1)
            sel = np.where(np.abs(pos[sel] - newp) <= np.abs(pos[(sel - 1).clip(0)] - newp), sel, (sel - 1).clip(0)); nul.append(st(sel))
        nul = np.array(nul); ob.append(o); nls.append(nul)
        sh = top[ci[top] == dom]; indep = False
        if len(sh) >= 2:
            rr = ld_r2(lc, g.atlas_variant.to_numpy()[sh]); iu2 = np.triu_indices(len(sh), 1); indep = bool(np.any(rr[iu2] < .5))
        pdom = (1 + (nul[:, 6] >= o[6]).sum()) / (len(nul) + 1)
        tb = g.iloc[top]; tb = tb[tb.ci == dom]
        rows.append(dict(locus=lc, k=k, dom_class=CL[dom], n_share=len(sh), indep=indep, frac=o[6], frac_null=nul[:, 6].mean(), p_dom=pdom,
                         bios=";".join(pd.Series(tb.bios).value_counts().index[:3]), genes=";".join(pd.Series(tb.gene).value_counts().index[:3])))
    ob, nls = np.array(ob), np.array(nls); out = {}
    for lab, j in [("(a) 최대 변화 부류 일치(전체)", 0), ("(a) 부류 일치(낮은 LD)", 1), ("(a) 부류 일치(높은 LD)", 2), ("(b) biosample 일치(전체)", 3), ("(c) 371 트랙 순서 상관(전체)", 4), ("(c) 순서 상관(낮은 LD)", 5)]:
        m = ~np.isnan(ob[:, j]); o_ = ob[m, j].mean(); Nn = np.nanmean(nls[m, :, j], 0); p = (1 + (Nn >= o_).sum()) / (len(Nn) + 1); out[lab] = p
        P(f"  {lab:28s} 관측 {o_:.4f} vs 귀무 {np.nanmean(Nn):.4f} | 단측 p {p:.4f} ({int(m.sum())} 좌위)")
    return out, pd.DataFrame(rows)
P("\n=== F1 (주): 좌위별 χ² 상위 1 % 의 최대 변화 세포 수렴 (RNA-seq; 거리 보존 평행이동 1,000회) ===")
o1, L = run(save=True)
v = "수렴 지지" if (o1["(a) 최대 변화 부류 일치(전체)"] < .05 and o1["(a) 부류 일치(낮은 LD)"] < .05) else ("태그·인접 효과" if o1["(a) 최대 변화 부류 일치(전체)"] < .05 else "불지지")
P(f"  → F1 판정: {v}")
P("\n=== F2: 좌위별 세포 × 기전 정의 가능성 ===")
L["q_dom"] = multipletests(L.p_dom, method="fdr_bh")[1]
L["definable"] = (L.n_share >= 3) & L.indep & (L.p_dom < .05); L["definable_fdr"] = L.definable & (L.q_dom < .1)
nd = int(L.definable.sum()); P(f"  정의 가능 좌위 (명목): {nd} / 75 (귀무 기대 약 {.05 * 75:.1f}; 이항 p {stats.binomtest(nd, 75, .05, alternative='greater').pvalue:.3g}) | BH-FDR < 0.1: {int(L.definable_fdr.sum())}")
T2 = pd.read_csv(RES/"e05_locus_table.csv", index_col=0)[["profile", "z_tilt_발현", "z_tilt_스플라이싱"]]
L = L.merge(T2, left_on="locus", right_index=True, how="left")
for r in L[L.definable].sort_values("p_dom").itertuples():
    P(f"  {r.locus:20s} {r.dom_class:8s} ({r.n_share}/{r.k} SNV, 비율 {r.frac:.2f} vs 귀무 {r.frac_null:.2f}, p {r.p_dom:.3f}, q {r.q_dom:.2f}) | 표본 {r.bios[:60]} | 유전자 {r.genes[:40]} | 1e-2 기전 {r.profile}")
P("  정의 가능 좌위의 주 부류 분포: " + ", ".join(f"{k} {v}" for k, v in L[L.definable].dom_class.value_counts().items()))
P("  (기술) 모든 좌위 주 부류 분포: " + ", ".join(f"{k} {v}" for k, v in L.dom_class.value_counts().items()))
P("\n=== 민감도: 상위 0.5 % / 2 % (F1 (a)) ===")
for f in [.005, .02]: P(f"  [상위 {f:.1%}]"); run(f, nperm=500)
# ---------- F3 스플라이싱 ----------
P("\n=== F3 (기술): 스플라이싱 최대 변화 세포 (값이 있는 SNV만) ===")
S = pd.concat([pd.read_parquet(f) for f in sorted(C.glob("spl_*.parquet")) if f.stat().st_size > 0 and len(pd.read_parquet(f))]).drop_duplicates("atlas_variant")
cols = [c for c in S.columns if c not in ("atlas_variant", "gene")]; Sv = S[cols].astype(float).to_numpy(); S["tr"] = np.nanargmax(np.nan_to_num(Sv, nan=-1), 1); S["cls"] = TS.cls.to_numpy()[S.tr]
DS = D.merge(S[["atlas_variant", "cls", "gene"]].rename(columns={"cls": "scls", "gene": "sgene"}), on="atlas_variant")
P(f"  스플라이싱 값 있는 SNV {len(DS):,} ({len(DS) / len(D):.1%}); 부류 분포: " + ", ".join(f"{c} {v:.0%}" for c, v in DS.scls.value_counts(normalize=True).head(6).items()))
D["pct"] = D.groupby("locus").chi2.rank(pct=True); DS = DS.merge(D[["atlas_variant", "pct"]], on="atlas_variant"); t5 = DS[DS.pct >= .95]; rr = []
for lc, g in t5.groupby("locus"):
    if len(g) >= 2: vc = g.scls.value_counts(); rr.append((lc, len(g), vc.index[0], vc.iloc[0], L.set_index("locus").dom_class.get(lc)))
P(f"  χ² 상위 5 % 안에 스플라이싱 SNV 2개 이상인 좌위 {len(rr)}개: 부류 일치(최빈 부류 ≥ 2개) {sum(r[3] >= 2 for r in rr)}개, 발현 주 부류와 같은 경우 {sum(r[2] == r[4] for r in rr)}개")
for r in rr: P(f"    {r[0]:20s} n {r[1]} | 스플라이싱 최빈 {r[2]} ({r[3]}) | 발현 주 부류 {r[4]}")
P("\n=== F1-스플라이싱 (부 검정; 수정 2): 스플라이싱 값 보유 SNV 만 ===")
cov = len(DS) / len(D); P(f"  보유 비율 {cov:.1%}")
if cov >= .30:
    Sfull = S.set_index("atlas_variant"); DSp = D.drop(columns=["tr", "gene", "bios", "cls", "ci"]).merge(S[["atlas_variant", "tr", "gene"]], on="atlas_variant").sort_values(["locus", "pos"]).reset_index(drop=True)
    DSp["bios"] = TS.biosample_name.to_numpy()[DSp.tr]; DSp["cls"] = TS.cls.to_numpy()[DSp.tr]; DSp["ci"] = DSp.cls.map({c: i for i, c in enumerate(CL)}).fillna(-1).astype(int)
    Rs = stats.rankdata(np.nan_to_num(Sfull.loc[DSp.atlas_variant, cols].astype(float).to_numpy(), nan=0), axis=1).astype(np.float32); Rs = (Rs - Rs.mean(1, keepdims=True)) / (Rs.std(1, keepdims=True) + 1e-9)
    D_rna, RK_rna = D, RK; D, RK = DSp, Rs
    os_, Ls = run(nperm=1000); D, RK = D_rna, RK_rna
    vs = "수렴 지지" if (os_["(a) 최대 변화 부류 일치(전체)"] < .05 and os_["(a) 부류 일치(낮은 LD)"] < .05) else ("태그·인접 효과" if os_["(a) 최대 변화 부류 일치(전체)"] < .05 else "불지지")
    P(f"  → F1-스플라이싱 판정: {vs}"); Ls.to_csv(RES/"e07_locus_celltype_splicing.csv", index=False)
else: P("  보유 비율 < 30 % → 실행하지 않음")
# ---------- GTEx 조직 대응 (기술) ----------
P("\n=== 보조: CS 인과 후보의 RNA 최대 변화 부류 vs GTEx eQTL 조직 (기술) ===")
# GTEx v8 single-tissue eQTL tissues of each credible-set lead were retrieved separately (GTEx portal, gtex_v8).
# Provide RES/"gtex_cs_lead_eqtl_tissues.csv" with columns lead (chr:pos:REF>ALT) and gtex_tissues (";"-separated tissueSiteDetailId).
CS = pd.read_csv(RES/"e05_cs.csv"); GT = pd.read_csv(RES/"gtex_cs_lead_eqtl_tissues.csv").fillna({"gtex_tissues": ""}).set_index("lead").gtex_tissues; rows = []
for av in CS.lead:
    if av not in set(D.atlas_variant): continue
    rows.append(dict(lead=av, cls=D.set_index("atlas_variant").cls.get(av), bios=D.set_index("atlas_variant").bios.get(av), gtex_tissues=GT.get(av, "")))
G = pd.DataFrame(rows); G.to_csv(RES/"e07_cs_gtex_tissue.csv", index=False)
big = {"뇌": r"Brain", "혈액·면역": r"Whole_Blood|Spleen|lymphocytes", "간": r"Liver"}
for lab, rx in big.items():
    has = G.gtex_tissues.str.contains(rx); P(f"  GTEx {lab} eQTL 있는 후보 {int(has.sum())}/{len(G)} | 그중 Atlas 최대 변화 부류: " + ", ".join(f"{k} {v}" for k, v in G[has].cls.value_counts().head(4).items()))
L.to_csv(RES/"e07_locus_celltype.csv", index=False); (RES/"e07_celltype.txt").write_text("\n".join(log), encoding="utf-8")
