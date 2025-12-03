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
library(lme4)

options(stringsAsFactors = FALSE)
DATA_DIR <- Sys.getenv("DATA_DIR", unset = "analysis/data")
print(DATA_DIR)

pre_train_cycle=read.csv(file.path(DATA_DIR, 'bart_train_cycle_pre.csv'))
# print(dim(pre_train_cycle))
pre_test_cycle=read.csv(file.path(DATA_DIR, 'bart_test_cycle_pre.csv'))
post_train_cycle=read.csv(file.path(DATA_DIR, 'bart_train_cycle_post.csv'))
post_test_cycle=read.csv(file.path(DATA_DIR, 'bart_test_cycle_post.csv'))

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

organize_probs<-function(pred, pre, x.test) {
  probs<-colMeans(pred$prob.test)
  N_test<-nrow(x.test)
  K<-pre$K
  tgrid<-pre$times
  print(dim(x.test))
  print(K)
  print(N_test)
  stopifnot(length(probs)==N_test*K)

  prob_mat<-matrix(probs, nrow=N_test, ncol=K, byrow=TRUE)
  colnames(prob_mat)<-paste0('t', tgrid)
  prob_df<-as.data.frame(prob_mat)
  prob_df
}


data_pre_cycle<-prepare_data(pre_train_cycle, pre_test_cycle)
data_post_cycle<-prepare_data(post_train_cycle, post_test_cycle)

pre.pre_cycle<-surv_pre(data_pre_cycle$times_tr, data_pre_cycle$delta_tr, data_pre_cycle$x.train, data_pre_cycle$x.test)
pre.post_cycle<-surv_pre(data_post_cycle$times_tr, data_post_cycle$delta_tr, data_post_cycle$x.train, data_post_cycle$x.test)

options(stringsAsFactors = FALSE)
ANALYSIS_PATH <- Sys.getenv("DATA_DIR", unset = "analysis/tables")

post.pre_cycle <- readRDS(file.path(ANALYSIS_PATH, 'bart_model_cycle_pre_best.rds'))
post.post_cycle <- readRDS(file.path(ANALYSIS_PATH, 'bart_model_cycle_post_best.rds'))

pred.pre_cycle<-predict(post.pre_cycle, newdata=pre.pre_cycle$tx.test)
pred.post_cycle<-predict(post.post_cycle, newdata=pre.post_cycle$tx.test)
df.pre_cycle<-organize_preds(pred.pre_cycle, pre.pre_cycle, data_pre_cycle$x.test)
df.post_cycle<-organize_preds(pred.post_cycle, pre.post_cycle, data_post_cycle$x.test)

df.prob.pre_cycle<-organize_probs(pred.pre_cycle, pre.pre_cycle, data_pre_cycle$x.test)
df.prob.post_cycle<-organize_probs(pred.post_cycle, pre.post_cycle, data_post_cycle$x.test)

target <- 8.39103033443728

post_test_cycle[post_test_cycle$female_age == target & post_test_cycle$time==5, ]


df.prob.post_cycle[178, ]
1-df.post_cycle[178, ]

1-df.pre_cycle[178, ]

# pre_test_cycle[c(2, 18), ]
# post_test_cycle[c(2, 18), ]
# patient 2 worse prognosis, patient 18 better prognosis 
pre_test_cycle[c(178, 18), ]

df.post_cycle[c(178, 18), ]

dim(pred.pre_cycle$surv.test)



# --- 1. Define Patient and Data Variables ---
options(stringsAsFactors = FALSE)
PLOT_PATH <- Sys.getenv("DATA_DIR", unset = "analysis/submission")

# ------------------
# Define patients and K (assuming K is length(tgrid) if not defined)
i1 <- 18 # better prognosis 
i2 <- 178 # worse prognosis 
tgrid <- pre.pre_cycle$times
# K is assumed to be the number of time points in tgrid, or defined elsewhere
K <- length(tgrid) 

# --- Start PNG device and set up 1 row, 2 columns layout ---
# Use par(mfrow) to split the plotting area
png(filename = file.path(PLOT_PATH, "use_cases.tiff"), 
    width = 12,        # Set the width in inches
    height = 6,        # Set the height in inches
    units = "in",      # Specify that width/height are in inches
    res = 1200)
par(mfrow = c(1, 2), # 1 row, 2 columns
    mar = c(5, 4, 4, 2) + 0.1) # Default margins, adjust if needed

# ==========================================================
# 1. Plot PRE-TREATMENT Survival Curves (Left Panel)
# ==========================================================

# --- Calculate Survival Curves for Pre-Treatment ---
# (i1)
S1_mean_pre <- df.pre_cycle[i1, ]
1-S1_mean_pre
cols_i1 <- ((i1 - 1) * K + 1):(i1 * K)
S1_draw_pre <- pred.pre_cycle$surv.test[, cols_i1, drop = FALSE]
S1_lo_pre <- apply(S1_draw_pre, 2, quantile, 0.025)
1-S1_lo_pre
S1_hi_pre <- apply(S1_draw_pre, 2, quantile, 0.975)
1-S1_hi_pre
# (i2)
S2_mean_pre <- df.pre_cycle[i2, ]
1-S2_mean_pre
cols_i2 <- ((i2 - 1) * K + 1):(i2 * K)
S2_draw_pre <- pred.pre_cycle$surv.test[, cols_i2, drop = FALSE]
S2_lo_pre <- apply(S2_draw_pre, 2, quantile, 0.025)

S2_hi_pre <- apply(S2_draw_pre, 2, quantile, 0.975)


# --- Plotting Pre-Treatment ---
plot(c(0, tgrid), c(1, S1_mean_pre), type = "s", lwd = 2, col = "blue",
     xlab = "Time", ylab = "Survival S(t|x)", ylim = c(0, 1),
     main = paste("Pre-Treatment Predictions: Couple", i1, "vs", i2))

# Add Patient i1's 95% bands (Blue, dashed)
lines(c(0, tgrid), c(1, S1_lo_pre), type = "s", lty = 2, col = "blue")
lines(c(0, tgrid), c(1, S1_hi_pre), type = "s", lty = 2, col = "blue")
# Add Patient i2's mean curve (Red)
lines(c(0, tgrid), c(1, S2_mean_pre), type = "s", lwd = 2, col = "red")
# Add Patient i2's 95% bands (Red, dashed)
lines(c(0, tgrid), c(1, S2_lo_pre), type = "s", lty = 2, col = "red")
lines(c(0, tgrid), c(1, S2_hi_pre), type = "s", lty = 2, col = "red")

# Custom Legend
legend("topright", bty = "n", cex = 0.8, # Added cex for smaller legend
       lwd = c(2, 2, 1, 1),
       lty = c(1, 1, 2, 2),
       col = c("blue", "red", "blue", "red"),
       legend = c(
         paste("Couple", i1, "Mean"),
         paste("Couple", i2, "Mean"),
         paste("Couple", i1, "95% Band"),
         paste("Couple", i2, "95% Band")
       ))

# ==========================================================
# 2. Plot POST-TREATMENT Survival Curves (Right Panel)
# ==========================================================

# --- Calculate Survival Curves for Post-Treatment ---
# (i1)
S1_mean_post <- df.post_cycle[i1, ]
1-S1_mean_post
S1_draw_post <- pred.post_cycle$surv.test[, cols_i1, drop = FALSE]
S1_lo_post <- apply(S1_draw_post, 2, quantile, 0.025)
1-S1_lo_post
S1_hi_post <- apply(S1_draw_post, 2, quantile, 0.975)
1-S1_hi_post
# (i2)
S2_mean_post <- df.post_cycle[i2, ]
1-S2_mean_post
S2_draw_post <- pred.post_cycle$surv.test[, cols_i2, drop = FALSE]
S2_lo_post <- apply(S2_draw_post, 2, quantile, 0.025)
S2_hi_post <- apply(S2_draw_post, 2, quantile, 0.975)

# --- Plotting Post-Treatment ---
plot(c(0, tgrid), c(1, S1_mean_post), type = "s", lwd = 2, col = "blue",
     xlab = "Time", ylab = "Survival S(t|x)", ylim = c(0, 1),
     main = paste("Post-Treatment Predictions: Couple", i1, "vs", i2))

# Add Patient i1's 95% bands (Blue, dashed)
lines(c(0, tgrid), c(1, S1_lo_post), type = "s", lty = 2, col = "blue")
lines(c(0, tgrid), c(1, S1_hi_post), type = "s", lty = 2, col = "blue")
# Add Patient i2's mean curve (Red)
lines(c(0, tgrid), c(1, S2_mean_post), type = "s", lwd = 2, col = "red")
# Add Patient i2's 95% bands (Red, dashed)
lines(c(0, tgrid), c(1, S2_lo_post), type = "s", lty = 2, col = "red")
lines(c(0, tgrid), c(1, S2_hi_post), type = "s", lty = 2, col = "red")

# Custom Legend
legend("topright", bty = "n", cex = 0.8, # Added cex for smaller legend
       lwd = c(2, 2, 1, 1),
       lty = c(1, 1, 2, 2),
       col = c("blue", "red", "blue", "red"),
       legend = c(
         paste("Couple", i1, "Mean"),
         paste("Couple", i2, "Mean"),
         paste("Couple", i1, "95% Band"),
         paste("Couple", i2, "95% Band")
       ))

# --- 6. Close the graphical device ---
dev.off()

# -------plot only one patient ------
# ----------------------------------------
# ========================================

png(filename = file.path(PLOT_PATH, "use_case.tiff"), 
    width = 12,        # Set the width in inches
    height = 6,        # Set the height in inches
    units = "in",      # Specify that width/height are in inches
    res = 1200)
par(mfrow = c(1, 2), # 1 row, 2 columns
    mar = c(5, 4, 4, 2) + 0.1) # Default margins, adjust if needed

# ==========================================================
# 1. Plot PRE-TREATMENT Survival Curves (Left Panel)
# ==========================================================

# --- Calculate Survival Curves for Pre-Treatment ---
# (i1)
S1_mean_pre <- df.pre_cycle[i1, ]
1-S1_mean_pre
cols_i1 <- ((i1 - 1) * K + 1):(i1 * K)
S1_draw_pre <- pred.pre_cycle$surv.test[, cols_i1, drop = FALSE]
S1_lo_pre <- apply(S1_draw_pre, 2, quantile, 0.025)
1-S1_lo_pre
S1_hi_pre <- apply(S1_draw_pre, 2, quantile, 0.975)
1-S1_hi_pre


# --- Plotting Pre-Treatment ---
plot(c(0, tgrid), c(1, S1_mean_pre), type = "s", lwd = 2, col = "blue",
     xlab = "Cycle", ylab = "Survival S(t|x)", ylim = c(0, 1))
    #  main = paste("Pre-Treatment Predictions: Couple", i1))

# Add Patient i1's 95% bands (Blue, dashed)
lines(c(0, tgrid), c(1, S1_lo_pre), type = "s", lty = 2, col = "blue")
lines(c(0, tgrid), c(1, S1_hi_pre), type = "s", lty = 2, col = "blue")


# Custom Legend
legend("topright", bty = "n", cex = 0.8,
       lwd = c(2, 1),
       lty = c(1, 2),
       col = c("blue", "blue"),
       legend = c(paste("Couple", i1, "mean"),
                  paste("Couple", i1, "95% band")))

# ==========================================================
# 2. Plot POST-TREATMENT Survival Curves (Right Panel)
# ==========================================================

# --- Calculate Survival Curves for Post-Treatment ---
# (i1)
S1_mean_post <- df.post_cycle[i1, ]
1-S1_mean_post
S1_draw_post <- pred.post_cycle$surv.test[, cols_i1, drop = FALSE]
S1_lo_post <- apply(S1_draw_post, 2, quantile, 0.025)
1-S1_lo_post
S1_hi_post <- apply(S1_draw_post, 2, quantile, 0.975)
1-S1_hi_post

# --- Plotting Post-Treatment ---
plot(c(0, tgrid), c(1, S1_mean_post), type = "s", lwd = 2, col = "blue",
     xlab = "Cycle", ylab = "Survival S(t|x)", ylim = c(0, 1))
    #  main = paste("Post-Treatment Predictions: Couple", i1))

# Add Patient i1's 95% bands (Blue, dashed)
lines(c(0, tgrid), c(1, S1_lo_post), type = "s", lty = 2, col = "blue")
lines(c(0, tgrid), c(1, S1_hi_post), type = "s", lty = 2, col = "blue")

# Custom Legend
legend("topright", bty = "n", cex = 0.8,
       lwd = c(2, 1),
       lty = c(1, 2),
       col = c("blue", "blue"),
       legend = c(paste("Couple", i1, "mean"),
                  paste("Couple", i1, "95% band")))

# --- 6. Close the graphical device ---
dev.off()









