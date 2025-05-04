# R methods for feature selection

required_packages <- c("party", "sandwich", "mvtnorm", "modeltools", "stats4", "strucchange", "zoo")
invisible(suppressPackageStartupMessages(lapply(required_packages, library, character.only = TRUE)))

suppressMessages(library(Boruta))
suppressMessages(library(knockoff))
suppressMessages(library(party))
suppressMessages(library(randomForest))
suppressMessages(library(stabs))
suppressMessages(library(vita))
suppressMessages(library(VSURF))
suppressMessages(library(xgboost))

run_boruta <- function(X, y, classifier = FALSE) {
	y <- as.vector(y)

	if (classifier) {
		y <- as.factor(y)
	}
	
	start_time <- Sys.time()
	boruta_fit <- Boruta(X, y)
	boruta_final <- suppressWarnings(TentativeRoughFix(boruta_fit))
	end_time <- Sys.time()
	runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
	
	selected_features <- which(boruta_final$finalDecision == "Confirmed")
	
	return(list(
		selected_features = selected_features,
		runtime = runtime
	))
}

run_hofner <- function(X, y, cutoff=0.75, PFER=1, mstop=100, baselearner="bbs", mc_cores=1) {
	y <- as.vector(y)
	data <- data.frame(X, Value = y)
	nms <- names(data)
	
	start_time <- Sys.time()
	fm <- as.formula(paste("Value ~ ", paste(nms[-length(nms)], collapse = "+")))
	mod <- gamboost(fm, data = data, control = boost_control(mstop = mstop), baselearner = baselearner)
	stab <- stabsel(mod, cutoff = cutoff, PFER = PFER, mc.cores = mc_cores)
	end_time <- Sys.time()
	runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
	
	selected_features <- stab$selected

	pattern <- paste0(baselearner, "\\(X(\\d+).*\\)")
	selected_indices <- as.numeric(sub(pattern, "\\1", names(selected_features)))
	selected_indices <- selected_indices[!is.na(selected_indices)]
	
	return(list(
	selected_features = selected_indices,
	runtime = runtime
	))
}

suppressMessages(library(knockoff))

run_knockoffs <- function(X, y, fdr_list, stat, mu=NULL, Sigma=NULL) {
	suppressMessages({
	suppressWarnings({
		start_time <- Sys.time()

		# Define the knockoff creation method and statistic
		if (!is.null(mu) && !is.null(Sigma)) {
		knockoffs <- function(X) knockoff::create.gaussian(X, mu, Sigma)
		} else {
		knockoffs <- function(X) knockoff::create.second_order(X)
		}
		
		# Ensure stat is dynamically called
		k_stat = function(X, Xk, y) do.call(stat, list(X, Xk, y, nfolds=5))

		# Compute the W statistics
		knockoff_result = knockoff.filter(X, y, knockoffs=knockoffs, statistic=k_stat, fdr=min(fdr_list))
		W = knockoff_result$statistic

		# Initialize scores with default value of 1
		scores <- rep(1, length(W))
		
		# Update scores with the smallest FDR at which each feature is selected
		for (fdr in fdr_list) {
		threshold = knockoff.threshold(W, fdr=fdr)
		selected = which(W >= threshold)
		scores[selected] = pmin(scores[selected], fdr)
		}

		end_time <- Sys.time()
		runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

		return(list(
		scores = scores,
		runtime = runtime
		))
	})
	})
}

run_kobt <- function(X, y, fdr_list, num_knockoffs = 2, knockoff_type = "shrink") {
	suppressMessages({
		suppressWarnings({

			# Source necessary scripts
			source("KOBT/R/generate.knockoff.R")
			source("KOBT/R/importance.score.R")
			source("KOBT/R/kobt.select.R")

			start_time <- Sys.time()

			# 1. Generate knockoffs
			Z_list <- generate.knockoff(X = X, type = knockoff_type, num = num_knockoffs)

			# 2. Compute SHAP scores for each knockoff copy
			score_list <- vector("list", length = num_knockoffs)
			for (i in seq_along(Z_list)) {
				Xk <- Z_list[[i]]
				X_aug <- cbind(X, Xk)
				dtrain <- xgboost::xgb.DMatrix(data = X_aug, label = y)
				fit <- xgboost::xgb.train(data = dtrain, nrounds = 20, verbose = 0)
				score_list[[i]] <- importance.score(fit = fit, Y = y, X = X_aug)$shap
			}

			# 3. Combine into score matrix (rows = replications, cols = original + knockoff features)
			score_matrix <- matrix(unlist(score_list), ncol = ncol(X) * 2, byrow = TRUE)

			# 4. Compute kobt-based selections
			scores <- rep(1, ncol(X))  # initialize with max FDR
			for (fdr in fdr_list) {
				selected <- kobt.select(score = score_matrix, fdr = fdr, type = "usual")
				scores[selected] <- pmin(scores[selected], fdr)
			}

			end_time <- Sys.time()
			runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

			return(list(
				scores = scores,
				runtime = runtime
			))
		})
	})
}

run_vita <- function(X, y, classifier = FALSE, alpha = 0) {
	y <- as.vector(y)
	if (classifier) {
	y <- as.factor(y)
	}

	start_time <- Sys.time()

	# compute cross-validated permutation variable importance
	cv_vi <- CVPVI(X, y)

	# compute p-values for cross-validated permutation variable importance measures
	vita_fit <- NTA(cv_vi$cv_varim)

	# extract selected features based on the cross-validated permutation variable importance
	selected_features <- which(vita_fit$pvalue <= alpha)

	end_time <- Sys.time()
	runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

	return(list(
	selected_features = selected_features,
	runtime = runtime
	))
}











