#!/usr/bin/bash -r

library(BART)
library(tidyverse)
library(haven)
library(lme4)
library(jtools)
library(ggstance)
library(effects)
library(eha)
library(discSurv)
library(brms)
library(Ecdat)
library(mgcv)
library(ggplot2)
library(grid)
library(survival)
library(caret)
# PATH<-"C:/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data"
options(stringsAsFactors = FALSE)
DATA_DIR <- Sys.getenv("DATA_DIR", unset = "analysis/data")
print(DATA_DIR)
pre_train_cycle=read.csv(file.path(DATA_DIR, 'bart_train_cycle_pre.csv'))
print(dim(pre_train_cycle))
pre_test_cycle=read.csv(file.path(DATA_DIR, 'bart_test_cycle_pre.csv'))
post_train_cycle=read.csv(file.path(DATA_DIR, 'bart_train_cycle_post.csv'))
post_test_cycle=read.csv(file.path(DATA_DIR, 'bart_test_cycle_post.csv'))

# print version of BART
packageVersion("BART")
# print version rBayesianOptimization
packageVersion("rBayesianOptimization")
create_folds<-function(data){
  data_df <- data %>% mutate(row_id = row_number())
  K=5
  # caret::createFolds returns a list of indices by default (stratified on event)
  folds <- createFolds(y = data_df$event, k = K, list = TRUE, returnTrain = FALSE)
  
  # convert to a vector of fold ids and add to df
  fold_ids <- integer(nrow(data_df))
  for (i in seq_along(folds)) {
    fold_ids[folds[[i]]] <- i
  }
  data_df <- data_df %>% mutate(outer_fold = fold_ids) 
  data_df
}
pre_train_df<-create_folds(pre_train_cycle)
post_train_df<-create_folds(post_train_cycle)
colnames(pre_train_df)
print(dim(pre_train_df))
print(dim(post_train_df))

prepare_data<-function(data_tr, data_te) {
  Xtr_df<-data_tr %>% select(-time, -event, -row_id, -outer_fold)
  Xte_df<-data_te %>% select(-time, -event, -row_id, -outer_fold)
  
  x.train<-model.matrix(~ . - 1, data=Xtr_df)
  x.test<-model.matrix(~ . - 1, data=Xte_df)
  
  times_tr<-as.integer(data_tr$time)
  times_te<-as.integer(data_te$time)
  
  delta_tr<-as.integer(data_tr$event)
  delta_te<-as.integer(data_te$event)
  
  list(
    x.train  = x.train,
    x.test   = x.test,
    times_tr = times_tr,
    times_te = times_te,
    delta_tr = delta_tr,
    delta_te = delta_te
  )
}

surv_pre<-function(times_tr, delta_train, x.train, x.test) {
  
  pre<-surv.pre.bart(times=times_tr, delta=delta_train, x.train=x.train, x.test=x.test)
  pre
}

run_bart <- function(x.train, x.test, times_tr, delta_tr, ntree=150, sparse=TRUE, k=2, power=2) {
  post <- surv.bart(
    x.train = x.train,
    x.test = x.test,
    times   = times_tr,
    delta   = delta_tr,
    ntree   = ntree,
    nskip   = 1000,
    ndpost  = 1000,
    keepevery = 5,
    sparse = sparse,
    a = 0.5,
    b = 1,
    k = k,
    power = power
  )
  post
}

# =================== hyperparameter tuning with Bayesian optimization ================
# =====================================================================================

organize_preds<-function(pred, pre, x.test) {
  preds<-pred$surv.test.mean
  N_test<-nrow(x.test)
  K<-pre$K
  tgrid<-pre$times
  print(dim(x.test))
  print(K)
  print(N_test)
  stopifnot(length(preds)==N_test*K)
  
  S_mat<-matrix(preds, nrow=N_test, ncol=K, byrow=TRUE)
  colnames(S_mat)<-paste0('t', tgrid)
  S_df<-as.data.frame(S_mat)
  S_df
}

fit_survbart_cv<-function(train, test, ntree = 150, k = 2.0, power = 2.0, sparse = TRUE){
  data<-prepare_data(train, test)
  print(dim(data$x.train))
  print(dim(data$x.test))
  print(length(data$times_tr))
  if (ncol(data$x.train) == 0 || ncol(data$x.test) == 0) return(NA)

  pre<-surv_pre(data$times_tr, data$delta_tr, data$x.train, data$x.test)
  if (sum(data$delta_tr) == 0 || nrow(train) < 2 || nrow(test) < 1) return(NA)

  post<-run_bart(data$x.train, data$x.test, data$times_tr, data$delta_tr, ntree=ntree, sparse=sparse, k=k, power=power)

  surv_test_matrix <- if (is.list(post$surv.test)) do.call(rbind, post$surv.test) else post$surv.test
  print(dim(surv_test_matrix))
  mean_surv_raw <- colMeans(surv_test_matrix)
  K_times <- length(post$times)
  # print(mean_surv_raw)
  mean_surv_matrix <- matrix(mean_surv_raw, nrow = K_times, byrow = FALSE)
  log_integrated_hazard <- -log(pmax(mean_surv_matrix, 1e-10))
  single_risk_score <- colSums(log_integrated_hazard)
  
  cindex_val <- concordance(Surv(data$times_te, data$delta_te) ~ single_risk_score, reverse = TRUE)$concordance
  return(cindex_val)
}

# df_train_k<-pre_train_df %>% filter(outer_fold != 1)
# df_test_k<-pre_train_df %>% filter(outer_fold == 1)

# fit_survbart_cv(df_train_k, df_test_k, ntree=150, k=2, power=2, base=0.95)


# ================ pre treatment model =============
# ================ Sparse = TRUE ===================

N_OUTER_SPLITS<-5
bart_full_cv_obj <- function(ntree, k, power) {
    ntree <- round(ntree)
    k <- k
    power <- power
    c_indices <- c()
    print(paste("ntree:", ntree, "k:", k, "power:", power))
    # Use the 5 outer folds (1 to N_OUTER_SPLITS) for CV on the full dataset
    for (k_fold in 1:N_OUTER_SPLITS) {
        df_train_k <- pre_train_df %>% filter(outer_fold != k_fold)
        df_test_k  <- pre_train_df %>% filter(outer_fold == k_fold)
        # print(dim(df_train_k))
        print(k_fold)
        cidx <- tryCatch({
          fit_survbart_cv(df_train_k, df_test_k,
                                  ntree = ntree, k = k, power = power)
        }, error = function(e) NA)
        print(cidx)
        c_indices <- c(c_indices, cidx)
    }
    print(cidx)
    mean_c <- mean(c_indices, na.rm = TRUE)
    return(list(Score = mean_c))
}

# bart_full_cv_obj(ntree=150, k=2, power=2)

BO_INIT_POINTS<-5
BO_N_ITER<-16
BO_results_final <- BayesianOptimization(
  FUN = bart_full_cv_obj,
  bounds = list(ntree = c(50, 200), k = c(2, 5), power = c(2, 4)),
  init_points = BO_INIT_POINTS,
  n_iter = BO_N_ITER,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.0,
  verbose = TRUE # Turn off verbose output inside the parallel worker
)
BO_results_final
print('pre sparse')
print(BO_results_final$Best_Value)


# ================ post treatment model =============
# ================ Sparse = TRUE ===================

N_OUTER_SPLITS<-5
bart_full_cv_obj <- function(ntree, k, power) {
    ntree <- round(ntree)
    k <- k
    power <- power
    c_indices <- c()
    print(paste("ntree:", ntree, "k:", k, "power:", power))
    # Use the 5 outer folds (1 to N_OUTER_SPLITS) for CV on the full dataset
    for (k_fold in 1:N_OUTER_SPLITS) {
        df_train_k <- post_train_df %>% filter(outer_fold != k_fold)
        df_test_k  <- post_train_df %>% filter(outer_fold == k_fold)
        # print(dim(df_train_k))
        print(k_fold)
        cidx <- tryCatch({
          fit_survbart_cv(df_train_k, df_test_k,
                                  ntree = ntree, k = k, power = power)
        }, error = function(e) NA)
        print(cidx)
        c_indices <- c(c_indices, cidx)
    }
    print(cidx)
    mean_c <- mean(c_indices, na.rm = TRUE)
    return(list(Score = mean_c))
}

# bart_full_cv_obj(ntree=150, k=2, power=2)

BO_INIT_POINTS<-5
BO_N_ITER<-16

BO_results_final_post <- BayesianOptimization(
  FUN = bart_full_cv_obj,
  bounds = list(ntree = c(50, 200), k = c(2, 5), power = c(2, 4)),
  init_points = BO_INIT_POINTS,
  n_iter = BO_N_ITER,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.0,
  verbose = TRUE # Turn off verbose output inside the parallel worker
)

BO_results_final_post
print('post sparse')
print(BO_results_final_post$Best_Value)


# ----------------- run model with the besst hyperparameters --------------
prepare_data<-function(data_tr, data_te) {
  Xtr_df<-data_tr %>% select(-time, -event)
  Xte_df<-data_te %>% select(-time, -event)
  
  x.train<-model.matrix(~ . - 1, data=Xtr_df)
  x.test<-model.matrix(~ . - 1, data=Xte_df)
  
  times_tr<-as.integer(data_tr$time)
  times_te<-as.integer(data_te$time)
  
  delta_tr<-as.integer(data_tr$event)
  delta_te<-as.integer(data_te$event)
  
  list(
    x.train  = x.train,
    x.test   = x.test,
    times_tr = times_tr,
    times_te = times_te,
    delta_tr = delta_tr,
    delta_te = delta_te
  )
}
data_pre_cycle<-prepare_data(pre_train_cycle, pre_test_cycle)
data_post_cycle<-prepare_data(post_train_cycle, post_test_cycle)

pre.pre_cycle<-surv_pre(data_pre_cycle$times_tr, data_pre_cycle$delta_tr, data_pre_cycle$x.train, data_pre_cycle$x.test)
pre.post_cycle<-surv_pre(data_post_cycle$times_tr, data_post_cycle$delta_tr, data_post_cycle$x.train, data_post_cycle$x.test)

post.pre_cycle<-run_bart(data_pre_cycle$x.train, data_pre_cycle$x.test, data_pre_cycle$times_tr, data_pre_cycle$delta_tr, ntree=round(BO_results_final$Best_Par['ntree']), sparse=TRUE, k=BO_results_final$Best_Par['k'], power=BO_results_final$Best_Par['power'])
post.post_cycle<-run_bart(data_post_cycle$x.train, data_post_cycle$x.test, data_post_cycle$times_tr, data_post_cycle$delta_tr, ntree=round(BO_results_final_post$Best_Par['ntree']), sparse=TRUE, k=BO_results_final_post$Best_Par['k'], power=BO_results_final_post$Best_Par['power'])

print(sort(post.post_cycle$varprob.mean, decreasing = TRUE))
print(sort(post.pre_cycle$varprob.mean, decreasing = TRUE))
varimp_post_cycle<-data.frame(
  # variable=colnames(data_pre_cycle$x.train),
  importance=sort(post.post_cycle$varprob.mean, decreasing = TRUE)
  )
varimp_pre_cycle<-data.frame(
  # variable=colnames(data_post_cycle$x.train),
  importance=sort(post.pre_cycle$varprob.mean, decreasing = TRUE)
)

pred.pre_cycle<-predict(post.pre_cycle, newdata=pre.pre_cycle$tx.test)
# pred.pre_transfer<-predict(post.pre_transfer, newdata=pre.pre_transfer$tx.test)
pred.post_cycle<-predict(post.post_cycle, newdata=pre.post_cycle$tx.test)
# pred.post_transfer<-predict(post.post_transfer, newdata=pre.post_transfer$tx.test)

df.pre_cycle<-organize_preds(pred.pre_cycle, pre.pre_cycle, data_pre_cycle$x.test)
# df.pre_transfer<-organize_preds(pred.pre_transfer, pre.pre_transfer, data_pre_transfer$x.test)
df.post_cycle<-organize_preds(pred.post_cycle, pre.post_cycle, data_post_cycle$x.test)
# df.post_transfer<-organize_preds(pred.post_transfer, pre.post_transfer, data_post_transfer$x.test)

dim(df.pre_cycle)

# ANALYSIS_PATH<-"C:/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables"
options(stringsAsFactors = FALSE)
ANALYSIS_PATH <- Sys.getenv("DATA_DIR", unset = "analysis/tables")
write.csv(df.post_cycle, file.path(ANALYSIS_PATH, 'preds_cycle_post_best.csv'), row.names = FALSE)

# save models 
saveRDS(post.post_cycle, file=file.path(ANALYSIS_PATH, 'bart_model_cycle_post_best.rds'))
varimp_post_cycle_file<-file.path(ANALYSIS_PATH, 'bart_varimp_cycle_post_best.csv')
write.csv(varimp_post_cycle, varimp_post_cycle_file, row.names = TRUE)


saveRDS(BO_results_final, file = file.path(ANALYSIS_PATH, 'bart_paras_cycle_pre_best.rds'))
saveRDS(BO_results_final_post, file = file.path(ANALYSIS_PATH, 'bart_paras_cycle_post_best.rds'))


# ##################################### SPARSE = FALSE #####################################


prepare_data<-function(data_tr, data_te) {
  Xtr_df<-data_tr %>% select(-time, -event, -row_id, -outer_fold)
  Xte_df<-data_te %>% select(-time, -event, -row_id, -outer_fold)
  
  x.train<-model.matrix(~ . - 1, data=Xtr_df)
  x.test<-model.matrix(~ . - 1, data=Xte_df)
  
  times_tr<-as.integer(data_tr$time)
  times_te<-as.integer(data_te$time)
  
  delta_tr<-as.integer(data_tr$event)
  delta_te<-as.integer(data_te$event)
  
  list(
    x.train  = x.train,
    x.test   = x.test,
    times_tr = times_tr,
    times_te = times_te,
    delta_tr = delta_tr,
    delta_te = delta_te
  )
}
fit_survbart_cv<-function(train, test, ntree = 150, k = 2.0, power = 2.0, sparse =FALSE){
  data<-prepare_data(train, test)
  print(dim(data$x.train))
  print(dim(data$x.test))
  print(length(data$times_tr))
  if (ncol(data$x.train) == 0 || ncol(data$x.test) == 0) return(NA)

  pre<-surv_pre(data$times_tr, data$delta_tr, data$x.train, data$x.test)
  if (sum(data$delta_tr) == 0 || nrow(train) < 2 || nrow(test) < 1) return(NA)

  post<-run_bart(data$x.train, data$x.test, data$times_tr, data$delta_tr, ntree=ntree, sparse=sparse, k=k, power=power)

  surv_test_matrix <- if (is.list(post$surv.test)) do.call(rbind, post$surv.test) else post$surv.test
  print(dim(surv_test_matrix))
  mean_surv_raw <- colMeans(surv_test_matrix)
  K_times <- length(post$times)
  # print(mean_surv_raw)
  mean_surv_matrix <- matrix(mean_surv_raw, nrow = K_times, byrow = FALSE)
  log_integrated_hazard <- -log(pmax(mean_surv_matrix, 1e-10))
  single_risk_score <- colSums(log_integrated_hazard)
  
  cindex_val <- concordance(Surv(data$times_te, data$delta_te) ~ single_risk_score, reverse = TRUE)$concordance
  return(cindex_val)
}


# ================== pre treatment model =============
# ====================================================
N_OUTER_SPLITS<-5
bart_full_cv_obj <- function(ntree, k, power) {
    ntree <- round(ntree)
    k <- k
    power <- power
    c_indices <- c()
    print(paste("ntree:", ntree, "k:", k, "power:", power))
    # Use the 5 outer folds (1 to N_OUTER_SPLITS) for CV on the full dataset
    for (k_fold in 1:N_OUTER_SPLITS) {
        df_train_k <- pre_train_df %>% filter(outer_fold != k_fold)
        df_test_k  <- pre_train_df %>% filter(outer_fold == k_fold)
        # print(dim(df_train_k))
        print(k_fold)
        cidx <- tryCatch({
          fit_survbart_cv(df_train_k, df_test_k,
                                  ntree = ntree, k = k, power = power)
        }, error = function(e) NA)
        print(cidx)
        c_indices <- c(c_indices, cidx)
    }
    print(cidx)
    mean_c <- mean(c_indices, na.rm = TRUE)
    return(list(Score = mean_c))
}

BO_INIT_POINTS<-5
BO_N_ITER<-16

BO_results_final_pre_sparse_false <- BayesianOptimization(
  FUN = bart_full_cv_obj,
  bounds = list(ntree = c(50, 200), k = c(2, 5), power = c(2, 4)),
  init_points = BO_INIT_POINTS,
  n_iter = BO_N_ITER,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.0,
  verbose = TRUE # Turn off verbose output inside the parallel worker
)

BO_results_final_pre_sparse_false
print('pre sparse false')
print(BO_results_final_pre_sparse_false$Best_Value)

# ================== post treatment model =============
# =====================================================
N_OUTER_SPLITS<-5
bart_full_cv_obj <- function(ntree, k, power) {
    ntree <- round(ntree)
    k <- k
    power <- power
    c_indices <- c()
    print(paste("ntree:", ntree, "k:", k, "power:", power))
    # Use the 5 outer folds (1 to N_OUTER_SPLITS) for CV on the full dataset
    for (k_fold in 1:N_OUTER_SPLITS) {
        df_train_k <- post_train_df %>% filter(outer_fold != k_fold)
        df_test_k  <- post_train_df %>% filter(outer_fold == k_fold)
        # print(dim(df_train_k))
        print(k_fold)
        cidx <- tryCatch({
          fit_survbart_cv(df_train_k, df_test_k,
                                  ntree = ntree, k = k, power = power)
        }, error = function(e) NA)
        print(cidx)
        c_indices <- c(c_indices, cidx)
    }
    print(cidx)
    mean_c <- mean(c_indices, na.rm = TRUE)
    return(list(Score = mean_c))
}

BO_results_final_post_sparse_false <- BayesianOptimization(
  FUN = bart_full_cv_obj,
  bounds = list(ntree = c(50, 200), k = c(2, 5), power = c(2, 4)),
  init_points = BO_INIT_POINTS,
  n_iter = BO_N_ITER,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.0,
  verbose = TRUE # Turn off verbose output inside the parallel worker
)
BO_results_final_post_sparse_false
print('post sparse false')
print(BO_results_final_post_sparse_false$Best_Value)
# save results
saveRDS(BO_results_final_post_sparse_false, file = file.path(ANALYSIS_PATH, "BART_params_post_sparse_false.RDS"))
saveRDS(BO_results_final_pre_sparse_false,  file = file.path(ANALYSIS_PATH, "BART_params_pre_sparse_false.RDS"))

# pre sparse false is the best 
# post sparse is the best 


# ########################################################################
# ----------------- run model with the besst hyperparameters --------------
# ########################################################################
post.pre_cycle<-run_bart(data_pre_cycle$x.train, data_pre_cycle$x.test, data_pre_cycle$times_tr, data_pre_cycle$delta_tr, ntree=round(BO_results_final_pre_sparse_false$Best_Par['ntree']), sparse=TRUE, k=BO_results_final_pre_sparse_false$Best_Par['k'], power=BO_results_final_pre_sparse_false$Best_Par['power'])

print(sort(post.pre_cycle$varprob.mean, decreasing = TRUE))

varimp_pre_cycle<-data.frame(
  # variable=colnames(data_post_cycle$x.train),
  importance=sort(post.pre_cycle$varprob.mean, decreasing = TRUE)
)

pred.pre_cycle<-predict(post.pre_cycle, newdata=pre.pre_cycle$tx.test)

df.pre_cycle<-organize_preds(pred.pre_cycle, pre.pre_cycle, data_pre_cycle$x.test)
dim(df.pre_cycle)
varimp_pre_cycle_file<-file.path(ANALYSIS_PATH, 'bart_varimp_cycle_pre_best.csv')
write.csv(varimp_pre_cycle, varimp_pre_cycle_file, row.names = TRUE)
saveRDS(post.pre_cycle, file=file.path(ANALYSIS_PATH, 'bart_model_cycle_pre_best.rds'))
write.csv(df.pre_cycle, file.path(ANALYSIS_PATH, 'preds_cycle_pre_best.csv'), row.names = FALSE)

best_params_pre=readRDS(file.path(ANALYSIS_PATH, 'BART_params_pre_sparse_false.RDS'))
best_params_psot=readRDS(file.path(ANALYSIS_PATH, 'bart_paras_cycle_post_best.rds'))
best_params_pre
best_params_psot
