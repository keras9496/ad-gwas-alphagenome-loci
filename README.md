# ad-gwas-alphagenome-loci

Analysis code for:

> Choi B, Jeong J. **Predicted transcript effects track association at Alzheimer's disease loci while cellular context remains unresolved.** bioRxiv (2026).

The study relates AlphaGenome Atlas variant-effect predictions to association strength within 75 Alzheimer's disease GWAS regions (Bellenguez et al. 2022, stage I summary statistics, GCST90027158; 111,446 common SNVs). It tests which predicted mechanism class tracks association and whether the credible-set candidates of a locus share a predicted cellular context.

## What this repository contains, and what it does not
- **Included**: the statistical analysis code, as run, with the date-stamped analysis stages described in the paper.
- **Not included**
  - Code that queries external services: the AlphaGenome / AlphaGenome Atlas API, Ensembl REST (VEP, gene and exon coordinates), the GTEx portal and the GWAS Catalog.
  - API keys.
  - Data: GWAS summary statistics, 1000 Genomes genotypes, Atlas scores, MPRA tables and intermediate files.
  - Manuscript-building scripts.
- **Four scripts mixed analysis with lookups.** In these, only the lookup block was replaced by a read of a locally stored table. Each is marked on line 1 with `# [public version]`.

| Script | Change |
|---|---|
| `scripts/s02_ld_tss.py`, `exp1e/scripts/f01_e8_loci.py` | Ensembl protein-coding TSS lookup → read of a cached `tss_cache.json` (`{"chr:start-end": [TSS, ...]}`) |
| `scripts/s31_rep25_prep.py` | GWAS Catalog region definition and an unused track-level Atlas query removed; expects `rep25/loci.csv` |
| `exp1e/scripts/e07_celltype_tests.py` | GTEx eQTL tissue lookup for credible-set leads → read of `results/gtex_cs_lead_eqtl_tissues.csv` |

## Layout
The folder structure mirrors the original project, because some scripts read parts of other scripts by relative path (for example, `e08`, `e14` and `make_figures2` reuse the cell-class rules in `e07`; `b02` runs `s15`).

```
scripts/        region definition, LD and covariates, SuSiE-RSS fine-mapping (s01, s02, s14, s15, s31)
exp1b/scripts/  SuSiE-RSS for the 25 replication regions (b02)
exp1e/scripts/  analyses of the paper (e02–e15) and the independent-loci replication (f01, f04)
exp1e/paper/    figures (make_figures, make_figures2) and figure deck (build_pptx)
```

## Analysis order
| Stage | Scripts | What it does |
|---|---|---|
| Regions, LD, covariates | `s01` → `s02`; `s31` | 50 genome-wide significant regions and 25 replication regions; 1000G EUR LD scores and LD matrices; TSS distance |
| Fine-mapping | `s14` → `s15`; `b02` | SuSiE-RSS with a uniform prior (no functional annotation) |
| Exploratory stage | `e02` | Aggregate variant-impact score vs association; mechanism-class attributions |
| Confirmatory stage | `e05` | Measurement validity, within-locus tests, highest-PIP vs correlated variants, genic-position control, GTEx QTL status, per-locus z (Table 1) |
| Cellular context | `e07` → `e08` | Top-set cell-class agreement; PIP-weighted entropy of predicted cellular labels |
| MPRA comparison | `e09` | Microglial MPRA allelic activity (Supplementary Note 1) |
| Direction profiles | `e11` | Risk-allele-oriented expression profiles of credible-set members vs the lead |
| Revision checks | `e14`, `e15` | Pairwise agreement by measure and LD stratum (Supplementary Table S4); secondary credible-set screen with exact null expectations |
| Independent sub-threshold loci | `f01` → `f04` | Pre-registered replication in loci not used in the study (added after the preprint) |
| Figures | `make_figures`, `make_figures2`, `build_pptx` | Main and supplementary figures |

The inputs that the lookup scripts produced (AVI model features, track-level RNA-seq and splicing quantiles, signed RNA-seq quantiles, VEP consequences, exon distances, GTEx eQTL/sQTL status) are read from a data root. The scripts assume it is `D:/AD_GWAS_Atlas_data/`. Change that path to run elsewhere. Random seeds are fixed in each script.

## Environment
- Python 3.13 (`requirements.txt`)
- R 4.1.3 with susieR 0.12.35
- plink2 v2.0.0-a.7.8

Console messages and comments are in Korean, as in the original analysis.

## Data sources
- GWAS: Bellenguez et al., Nat Genet 2022, GWAS Catalog GCST90027158
- LD reference: 1000 Genomes Project phase 3, GRCh38, unrelated EUR samples
- Variant-effect predictions: AlphaGenome Atlas, obtained through the AlphaGenome API under its terms of use
- Annotations: Ensembl release 116 (REST API, accessed September 2026); GTEx v8 (GTEx portal)
- MPRA: allelic activity tables from the publications cited in the paper
