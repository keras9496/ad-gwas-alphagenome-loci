# 2단계 SuSiE-RSS. 사용: Rscript s15_susie.R P0|P1|P2|P1_noout  (P1/P2 는 work/s2/prior_<P>/<locus>.csv 의 prior 열 사용)
# 출력: work/s2/out_<P>/<locus>_pip.csv, <locus>_cs.csv (P0 는 kriging 이상치 <locus>_krig.csv 도)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages(library(susieR))
args <- commandArgs(TRUE); PR <- args[1]; W <- "D:/AD_GWAS_Atlas_data/work/s2"
out <- file.path(W, paste0("out_", PR)); dir.create(out, showWarnings = FALSE)
loci <- if (length(args) > 1) args[-1] else sub("_z.csv$", "", list.files(W, pattern = "_z.csv$"))
for (lc in loci) {
  if (file.exists(file.path(out, paste0(lc, "_pip.csv")))) next
  zz <- read.csv(file.path(W, paste0(lc, "_z.csv"))); p <- nrow(zz); n <- zz$n[1]
  R <- matrix(readBin(file.path(W, paste0(lc, "_R.bin")), "numeric", size = 4, n = p * p), p, p); R <- (R + t(R)) / 2; diag(R) <- 1
  keep <- rep(TRUE, p); pw <- NULL
  if (PR != "P0") {
    pf <- read.csv(file.path(W, paste0("prior_", sub("_noout", "", PR)), paste0(lc, ".csv")))
    stopifnot(all(pf$atlas_variant == zz$atlas_variant)); pw <- pf$prior
  }
  if (grepl("_noout", PR)) {
    kf <- file.path(W, "out_P0", paste0(lc, "_krig.csv")); if (file.exists(kf)) { k <- read.csv(kf); keep <- !(zz$atlas_variant %in% k$atlas_variant[k$outlier]) }
  }
  run <- function(Rm, lam) {
    Rm2 <- if (lam > 0) (1 - lam) * Rm + lam * diag(nrow(Rm)) else Rm
    susie_rss(zz$z[keep], Rm2, n = n, L = 10, coverage = 0.95, min_abs_corr = 0.5, estimate_residual_variance = FALSE,
              prior_weights = if (is.null(pw)) NULL else pw[keep] / sum(pw[keep]), max_iter = 500)
  }
  Rk <- R[keep, keep]; lam <- 0
  fit <- tryCatch(run(Rk, 0), error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) { lam <- 0.001; fit <- run(Rk, lam) }
  pip <- data.frame(atlas_variant = zz$atlas_variant[keep], pip = fit$pip, cs = NA_integer_)
  cs <- fit$sets$cs; rows <- list()
  if (length(cs)) for (i in seq_along(cs)) {
    nm <- names(cs)[i]; idx <- cs[[i]]; pip$cs[idx] <- as.integer(sub("L", "", nm))
    rows[[i]] <- data.frame(locus = lc, cs = as.integer(sub("L", "", nm)), size = length(idx), max_pip = max(fit$pip[idx]),
                            purity = fit$sets$purity[i, "min.abs.corr"], lead = zz$atlas_variant[keep][idx[which.max(fit$pip[idx])]])
  }
  write.csv(pip, file.path(out, paste0(lc, "_pip.csv")), row.names = FALSE)
  write.csv(if (length(rows)) do.call(rbind, rows) else data.frame(locus = lc, cs = NA, size = 0, max_pip = NA, purity = NA, lead = NA),
            file.path(out, paste0(lc, "_cs.csv")), row.names = FALSE)
  if (PR == "P0") {
    kr <- kriging_rss(zz$z, R, n = n)$conditional_dist
    write.csv(data.frame(atlas_variant = zz$atlas_variant, z = zz$z, z_std_diff = kr$z_std_diff, logLR = kr$logLR,
                         outlier = kr$logLR > 2 & abs(zz$z) > 2), file.path(out, paste0(lc, "_krig.csv")), row.names = FALSE)
  }
  cat(sprintf("%s %s: p=%d, CS %d, lambda %.3f, converged %s\n", PR, lc, sum(keep), length(cs), lam, fit$converged))
}
