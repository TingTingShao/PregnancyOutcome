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

# PATH<-"C:/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data"
options(stringsAsFactors = FALSE)
DATA_DIR <- Sys.getenv("DATA_DIR", unset = "analysis/data")
print(DATA_DIR)

# --------- data with transformed features -----
pre_train_transformed=read.csv(file.path(DATA_DIR, 'bart_data_tr_pre_transformed_ml_refit.csv'))
pre_test_transformed=read.csv(file.path(DATA_DIR, 'bart_data_te_pre_transformed_ml_refit.csv'))
post_train_transformed=read.csv(file.path(DATA_DIR, 'bart_data_tr_post_transformed_ml_refit.csv'))
post_test_transformed=read.csv(file.path(DATA_DIR, 'bart_data_te_post_transformed_ml_refit.csv'))

# ----- data with no transformed features but same number of features used in mclernon
pre_train=read_csv(file.path(DATA_DIR, 'bart_data_tr_pre_feats_ml_refit.csv'))
dim(pre_train)
colnames(pre_train)
pre_test=read.csv(file.path(DATA_DIR, 'bart_data_te_pre_feats_ml_refit.csv'))
dim(pre_test)
colnames(pre_test)
post_train=read.csv(file.path(DATA_DIR, 'bart_data_tr_post_feats_ml_refit.csv'))
dim(post_train)
post_test=read.csv(file.path(DATA_DIR, 'bart_data_te_post_feats_ml_refit.csv'))
dim(post_test)
#
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

surv_pre<-function(times_tr, delta_train, x.train, x.test) {
  
  pre<-surv.pre.bart(times=times_tr, delta=delta_train, x.train=x.train, x.test=x.test)
  pre
}

run_bart<-function(x.train, times_tr, delta_tr) {
  
  post <- surv.bart(
    x.train = x.train,
    times   = times_tr,
    delta   = delta_tr,
    ntree   = 100,
    nskip   = 1000,
    ndpost  = 1000,
    keepevery = 5
  )
  post
}

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


# -------- prepare data --------
pre_transformed<-prepare_data(pre_train_transformed, pre_test_transformed)
post_transformed<-prepare_data(post_train_transformed, post_test_transformed)
# ext_pre_transformed<-prepare_data(ext_pre_transformed, ext_pre_transformed)
# ext_post_transformed<-prepare_data(ext_post_transformed, ext_post_transformed)

pre_data<-prepare_data(pre_train, pre_test)
# pre_data

post_data<-prepare_data(post_train, post_test)
# ext_pre<-prepare_data(ext_pre, ext_pre)
# ext_post<-prepare_data(ext_post, ext_post)

# ----- surv data ----
pre.pre_transformed<-surv_pre(pre_transformed$times_tr, pre_transformed$delta_tr, pre_transformed$x.train, pre_transformed$x.test)
pre.post_transformed<-surv_pre(post_transformed$times_tr, post_transformed$delta_tr, post_transformed$x.train, post_transformed$x.test)
# pre.ext_pre_transformed<-surv_pre(ext_pre_transformed$times_tr, ext_pre_transformed$delta_tr, ext_pre_transformed$x.train, ext_pre_transformed$x.test)
# pre.ext_post_transformed<-surv_pre(ext_post_transformed$times_tr, ext_post_transformed$delta_tr, ext_post_transformed$x.train, ext_post_transformed$x.test)

pre.pre_data<-surv_pre(pre_data$times_tr, pre_data$delta_tr, pre_data$x.train, pre_data$x.test)
pre.post_data<-surv_pre(post_data$times_tr, post_data$delta_tr, post_data$x.train, post_data$x.test)
# pre.ext_pre<-surv_pre(ext_pre$times_tr, ext_pre$delta_tr, ext_pre$x.train, ext_pre$x.test)
# pre.ext_post<-surv_pre(ext_post$times_tr, ext_post$delta_tr, ext_post$x.train, ext_post$x.test)

# ---- run model ----
post.pre_transformed<-run_bart(pre_transformed$x.train, pre_transformed$times_tr, pre_transformed$delta_tr)
post.post_transformed<-run_bart(post_transformed$x.train, post_transformed$times_tr, post_transformed$delta_tr)
post.pre_data<-run_bart(pre_data$x.train, pre_data$times_tr, pre_data$delta_tr)
post.post_data<-run_bart(post_data$x.train, post_data$times_tr, post_data$delta_tr)

# ---- get predictions ----
pred.pre_cycle_transformed<-predict(post.pre_transformed, newdata=pre.pre_transformed$tx.test)
pred.post_cycle_transformed<-predict(post.post_transformed, newdata=pre.post_transformed$tx.test)


pred.pre_cycle<-predict(post.pre_data, newdata=pre.pre_data$tx.test)
pred.post_cycle<-predict(post.post_data, newdata=pre.post_data$tx.test)


# ---- organize predictions ----
df.pre_cycle_transformed<-organize_preds(pred.pre_cycle_transformed, pre.pre_transformed, pre_transformed$x.test)
df.post_cycle_transformed<-organize_preds(pred.post_cycle_transformed, pre.post_transformed, post_transformed$x.test)
df.pre_cycle<-organize_preds(pred.pre_cycle, pre.pre_data, pre_data$x.test)
df.post_cycle<-organize_preds(pred.post_cycle, pre.post_data, post_data$x.test)


# ---- save predictions ----
ANALYSIS_PATH<-"C:/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables"
options(stringsAsFactors = FALSE)
ANALYSIS_PATH <- Sys.getenv("DATA_DIR", unset = "analysis/tables")
write.csv(df.pre_cycle_transformed, file.path(ANALYSIS_PATH, "BART_pre_cycle_transformed_performance_feats_ml_refit.csv"), row.names = FALSE)
write.csv(df.post_cycle_transformed, file.path(ANALYSIS_PATH, "BART_post_cycle_transformed_performance_feats_ml_refit.csv"), row.names = FALSE)
write.csv(df.pre_cycle, file.path(ANALYSIS_PATH, "BART_pre_cycle_performance_feats_ml_refit.csv"), row.names = FALSE)
write.csv(df.post_cycle, file.path(ANALYSIS_PATH, "BART_post_cycle_performance_feats_ml_refit.csv"), row.names = FALSE)


