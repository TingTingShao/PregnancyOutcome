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
# PATH<-"C:/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data"
options(stringsAsFactors = FALSE)
DATA_DIR <- Sys.getenv("DATA_DIR", unset = "analysis/data")
print(DATA_DIR)
pre_train_cycle=read.csv(file.path(DATA_DIR, 'bart_train_cycle_pre.csv'))
print(dim(pre_train_cycle))
pre_test_cycle=read.csv(file.path(DATA_DIR, 'bart_test_cycle_pre.csv'))
post_train_cycle=read.csv(file.path(DATA_DIR, 'bart_train_cycle_post.csv'))
post_test_cycle=read.csv(file.path(DATA_DIR, 'bart_test_cycle_post.csv'))

sum(pre_train_cycle$time)
sum(pre_test_cycle$time)

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

data_pre_cycle<-prepare_data(pre_train_cycle, pre_test_cycle)
data_post_cycle<-prepare_data(post_train_cycle, post_test_cycle)

pre.pre_cycle<-surv_pre(data_pre_cycle$times_tr, data_pre_cycle$delta_tr, data_pre_cycle$x.train, data_pre_cycle$x.test)
pre.post_cycle<-surv_pre(data_post_cycle$times_tr, data_post_cycle$delta_tr, data_post_cycle$x.train, data_post_cycle$x.test)


options(stringsAsFactors = FALSE)
ANALYSIS_PATH <- Sys.getenv("DATA_DIR", unset = "analysis/tables")
# read models
post.pre_cycle <- readRDS(file.path(ANALYSIS_PATH, 'bart_model_cycle_pre_best.rds'))
post.post_cycle <- readRDS(file.path(ANALYSIS_PATH, 'bart_model_cycle_post_best.rds'))


# --------------------------- plot the partial dependancy plot ---------------------
options(stringsAsFactors = FALSE)
PLOT_PATH <- Sys.getenv("DATA_DIR", unset = "analysis/submission")
partial_dependence_bart<-function(train, test, model){
  data_prepared<-prepare_data(train, test)
  pre<-surv_pre(data_prepared$times_tr, data_prepared$delta_tr, data_prepared$x.train, data_prepared$x.test)
  # post<-run_bart(data_prepared$x.train, data_prepared$times_tr, data_prepared$delta_tr)
  pred<-predict(model, newdata=pre$tx.test)
  df<-organize_preds(pred, pre, data_prepared$x.test)
  # mean survival at each time point
  mean_surv<-colMeans(df)
  return(mean_surv)
}

pdp_list_to_df <- function(pdp_results, time_name_pattern = "^t", prefix = "Age-") {
  stopifnot(is.list(pdp_results), length(pdp_results) > 0)

  # fallback names if missing
  names <- names(pdp_results)
  if (is.null(names)) names <- as.character(seq_along(pdp_results))

  do.call(rbind, lapply(seq_along(pdp_results), function(i) {
    v <- pdp_results[[i]]
    data.frame(
      time = as.numeric(sub(time_name_pattern, "", names(v))),
      S    = as.numeric(v),
      feature  = paste0(prefix, names[i]),
      stringsAsFactors = FALSE
    )
  }))
}


# -------pretreatment cycle model partial dependence on female age, infertility duration, ovulation problem, male pathology -----
sd_age<-4.73
ages<-c(20, 25, 30, 35, 40)
# --- partial dependence plots ---
# empty list to store results
pdp_results_pre_age<-list()
for (age in ages){
  print(paste("Calculating partial dependence for age:", age))
  # create modified test set with fixed age
  test_modified<-pre_test_cycle
  test_modified$female_age<-age/sd_age

  mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
  pdp_results_pre_age[[as.character(age)]]<-mean_surv
}

print(pdp_results_pre_age)

# ------- pretreatment cycle model infertility duration -----
sd_infertility_duration<-2.23
infertility_durations<-c(1, 2, 3, 4, 5, 6)
pdp_results_pre_infertility_duration<-list()
for (infertility_duration in infertility_durations){
  print(paste("Calculating partial dependence for infertility_duration:", infertility_duration))
  # create modified test set with fixed age
  test_modified<-pre_test_cycle
  test_modified$infertility_duration<-infertility_duration/sd_infertility_duration

  mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
  pdp_results_pre_infertility_duration[[as.character(infertility_duration)]]<-mean_surv
}
print(pdp_results_pre_infertility_duration)

# ------- pretreatment cycle model ovulation problem -----
implantation_problems<-c(0, 1)
pdp_results_pre_implantation<-list()
pre_test_cycle$implantation_problem<-as.numeric(pre_test_cycle$implantation_problem)
for (impl in implantation_problems){
  print(paste("Calculating partial dependence for implantation_problem:", impl))
  # create modified test set with fixed age
  test_modified<-pre_test_cycle
  test_modified$implantation_problem<-impl

  mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
  pdp_results_pre_implantation[[as.character(impl)]]<-mean_surv
}
print(pdp_results_pre_implantation)

# ------- pretreatment cycle model infertility type -----
male_pathologys<-c(0, 1)
pdp_results_pre_male_pathology<-list()
pre_test_cycle$male_pathology<-as.numeric(pre_test_cycle$male_pathology)
for (mal in male_pathologys){
  print(paste("Calculating partial dependence for male_pathology:", mal))
  # create modified test set with fixed age
  test_modified<-pre_test_cycle
  test_modified$male_pathology<-mal
  mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
  pdp_results_pre_male_pathology[[as.character(mal)]]<-mean_surv
}
print(pdp_results_pre_male_pathology)
# ------- pretreatment cycle model FSH -----
# FSH_sd<-2.65
# FSHs<-c(3, 5, 7, 9)
# pdp_results_pre_FSH<-list()
# for (FSH in FSHs){
#   print(paste("Calculating partial dependence for FSH:", FSH))
#   # create modified test set with fixed age
#   test_modified<-pre_test_cycle
#   test_modified$FSH<-FSH/FSH_sd

#   mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
#   pdp_results_pre_FSH[[as.character(FSH)]]<-mean_surv
# }
# pdp_results_pre_FSH


# ---- plot the partial dependence for pretreatment model 
df_pre_age <- pdp_list_to_df(pdp_results_pre_age, prefix = "Age-")
df_pre_duration <- pdp_list_to_df(pdp_results_pre_infertility_duration, prefix = "Duration-")
df_pre_implantation <- pdp_list_to_df(pdp_results_pre_implantation, prefix = "Implantation-")
df_pre_male_pathology <- pdp_list_to_df(pdp_results_pre_male_pathology, prefix = "MalePathology-")
plot1<-ggplot(df_pre_age, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Female Age",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) ) 
plot2<-ggplot(df_pre_duration, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Infertility Duration",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) )
plot3<-ggplot(df_pre_implantation, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Implantation Problem",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) )
plot4<-ggplot(df_pre_male_pathology, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Male Pathology",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) )

write.csv(df_pre_age, file.path(ANALYSIS_PATH, 'pdp_pre_age.csv'), row.names = FALSE)
write.csv(df_pre_duration, file.path(ANALYSIS_PATH, 'pdp_pre_duration.csv'), row.names = FALSE)
write.csv(df_pre_implantation, file.path(ANALYSIS_PATH, 'pdp_pre_implantation.csv'), row.names = FALSE)
write.csv(df_pre_male_pathology, file.path(ANALYSIS_PATH, 'pdp_pre_male_pathology.csv'), row.names = FALSE)

png(filename=file.path(PLOT_PATH, "partial_dependence_plots_pre.tiff"), width = 1200, height = 600)
grid.newpage()
pushViewport(viewport(layout=grid.layout(2,2)))
vplayout<-function(x,y){
  viewport(layout.pos.row=x, layout.pos.col=y)
}
print(plot1, vp=vplayout(1,1))
print(plot2, vp=vplayout(1,2))
print(plot3, vp=vplayout(2,1))
print(plot4, vp=vplayout(2,2))
dev.off()

# -------pretreatment cycle model interaction between female age and infertility duration -----
sd_age<-4.73
ages<-c(25, 30, 35, 40)
sd_infertility_duration<-2.23
infertility_durations<-c(1, 2, 3, 4, 5, 6)
pdp_results_pre_age_duration<-list()
for (age in ages){
  for (infertility_duration in infertility_durations){
    test_modified<-pre_test_cycle
    test_modified$female_age<-age/sd_age
    test_modified$infertility_duration<-infertility_duration/sd_infertility_duration
    mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
    # store results
    pdp_results_pre_age_duration[[paste0("age", age, "_duration", infertility_duration)]]<-mean_surv
  }
}

implantation_problems<-c(0, 1)
pdp_results_pre_age_implantation<-list()
for (age in ages){
  for (impl in implantation_problems){
    test_modified<-pre_test_cycle
    test_modified$female_age<-age/sd_age
    test_modified$implantation_problem<-impl
    mean_surv<-partial_dependence_bart(pre_train_cycle, test_modified, model=post.pre_cycle)
    # store results
    pdp_results_pre_age_implantation[[paste0("age", age, "_impl", impl)]]<-mean_surv
  }
}
df_pre_age_duration <- pdp_list_to_df(pdp_results_pre_age_duration, prefix = "")
df_pre_age_implantation <- pdp_list_to_df(pdp_results_pre_age_implantation, prefix = "")
write.csv(df_pre_age_duration, file.path(ANALYSIS_PATH, 'pdp_pre_age_duration.csv'), row.names = FALSE)
write.csv(df_pre_age_implantation, file.path(ANALYSIS_PATH, 'pdp_pre_age_implantation.csv'), row.names = FALSE)

plot1<-ggplot(df_pre_age_duration, aes(time, S, color = feature)) +
  geom_line(size = 1) +
  labs(title = "Infertility Duration and Age",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) ) 
plot2<-ggplot(df_pre_age_implantation, aes(time, S, color = feature)) +
  geom_line(size = 1) +
  labs(title = "Implantation Problem and Age",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) ) 

png(filename=file.path(PLOT_PATH, "partial_dependence_plots_interaction_pre.tiff"), width = 1200, height = 600)
grid.newpage()
pushViewport(viewport(layout=grid.layout(1,2)))
vplayout<-function(x,y){
  viewport(layout.pos.row=x, layout.pos.col=y)
}
print(plot1, vp=vplayout(1, 1))
print(plot2, vp=vplayout(1, 2))
dev.off()


# ---------pretreatment cycle model embryo utility rate, n.2pn, female age, EM_last_measure-----

# --------- posttreatment cycle model embryo utility rate, n.2pn, female age, EM_last_measure-----

# --- partial dependence plots ---
# empty list to store results
sd_eur<-0.33
eurs<-c(0, 0.2, 0.4, 0.6, 0.8, 1)
pdp_results_post_eur<-list()
for (eur in eurs){
  print(paste("Calculating partial dependence for eur:", eur))
  # create modified test set with fixed eur
  test_modified<-post_test_cycle
  test_modified$embryo_utility_rate<-eur/sd_eur

  mean_surv<-partial_dependence_bart(post_train_cycle, test_modified, model=post.post_cycle)
  pdp_results_post_eur[[as.character(eur)]]<-mean_surv
}
print(pdp_results_post_eur)



# n.2pn
sd_2pn<-4.03
two_pns <-c(0, 2, 4, 6, 8)
pdp_results_post_2pn<-list()
for (n2pn in two_pns){
  print(paste("Calculating partial dependence for n.2pn:", n2pn))
  # create modified test set with fixed n.2pn
  test_modified<-post_test_cycle
  test_modified$n.2pn<-n2pn/sd_2pn

  mean_surv<-partial_dependence_bart(post_train_cycle, test_modified, model=post.post_cycle)
  pdp_results_post_2pn[[as.character(n2pn)]]<-mean_surv
}
pdp_results_post_2pn

# female age 
sd_age<-4.73
ages<-c(20, 25, 30, 35, 40)
# --- partial dependence plots ---
# empty list to store results
pdp_results_post_age<-list()
for (age in ages){
  print(paste("Calculating partial dependence for age:", age))
  # create modified test set with fixed age
  test_modified<-post_test_cycle
  test_modified$female_age<-age/sd_age

  mean_surv<-partial_dependence_bart(post_train_cycle, test_modified, model=post.post_cycle)
  pdp_results_post_age[[as.character(age)]]<-mean_surv
}

print(pdp_results_post_age)

#  EM last measure
sd_EM<-2.24
EMs<-c(0, 3, 5, 7, 9, 11, 13)
pdp_results_post_EM<-list()
for (EM in EMs){
  print(paste("Calculating partial dependence for EM_last_measure:", EM))
  # create modified test set with fixed EM_last_measure
  test_modified<-post_test_cycle
  test_modified$EM_last_measure<-EM/sd_EM

  mean_surv<-partial_dependence_bart(post_train_cycle, test_modified, model=post.post_cycle)
  pdp_results_post_EM[[as.character(EM)]]<-mean_surv
}

print(pdp_results_post_EM)

df_post_eur <- pdp_list_to_df(pdp_results_post_eur, prefix = "EUR-")
df_post_2pn <- pdp_list_to_df(pdp_results_post_2pn, prefix = "n.2pn-")
df_post_age <- pdp_list_to_df(pdp_results_post_age, prefix = "Female age-")
df_post_em <- pdp_list_to_df(pdp_results_post_EM, prefix = "EM-")

write.csv(df_post_eur, file.path(ANALYSIS_PATH, 'pdp_post_eur.csv'), row.names = FALSE)
write.csv(df_post_2pn, file.path(ANALYSIS_PATH, 'pdp_post_2pn.csv'), row.names = FALSE)
write.csv(df_post_age, file.path(ANALYSIS_PATH, 'pdp_post_age.csv'), row.names = FALSE)
write.csv(df_post_em, file.path(ANALYSIS_PATH, 'pdp_post_em.csv'), row.names = FALSE)
# read csv
df_post_eur<-read.csv(file.path(ANALYSIS_PATH, 'pdp_post_eur.csv'))
df_post_2pn<-read.csv(file.path(ANALYSIS_PATH, 'pdp_post_2pn.csv'))
df_post_age<-read.csv(file.path(ANALYSIS_PATH, 'pdp_post_age.csv'))
df_post_em<-read.csv(file.path(ANALYSIS_PATH, 'pdp_post_em.csv'))
head(df_post_2pn)
head(df_post_2pn)
head(df_post_age) 
df_post_age <- df_post_age %>%
  mutate(feature = sub("^Female age", "Age", feature))
plot1<-ggplot(df_post_eur, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Embryo Utilization Rate",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) ) 
plot2<-ggplot(df_post_2pn, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "The number of 2pn zygotes",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) )
plot3<-ggplot(df_post_age, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Female Age",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) )
plot4<-ggplot(df_post_em, aes(time, S, color = feature, linetype = feature)) +
  geom_line(size = 1) +
  labs(title = "Endometrial thickness",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) )



png(filename=file.path(PLOT_PATH, "partial_dependence_plots_post.tiff"), width = 1200, height = 600)
grid.newpage()
pushViewport(viewport(layout=grid.layout(2,2)))
vplayout<-function(x,y){
  viewport(layout.pos.row=x, layout.pos.col=y)
}
print(plot1, vp=vplayout(1,1))
print(plot3, vp=vplayout(1,2))
print(plot2, vp=vplayout(2,1))
print(plot4, vp=vplayout(2,2))
dev.off()

# ----------------- 2D partial plot ------------------
sd_eur<-0.33
eurs<-c(0, 0.4, 0.6, 0.8, 1)
sd_age<-4.73
ages<-c(25, 30, 35, 40)
pdp_results_post_age_eur<-list()
for (age in ages){
  for (eur in eurs){
    print(paste("Calculating partial dependence for age:", age, "and eur:", eur))
    # create modified test set with fixed age
    test_modified<-post_test_cycle
    test_modified$female_age<-age/sd_age
    test_modified$embryo_utility_rate<-eur/sd_eur

    mean_surv<-partial_dependence_bart(post_train_cycle, test_modified, model=post.post_cycle)
    pdp_results_post_age_eur[[paste0("Age", age, "_EUR", eur)]]<-mean_surv
  }
}

df_post_age_eur <- pdp_list_to_df(pdp_results_post_age_eur, prefix = "")


sd_2pn<-4.03
two_pns <-c(0, 2, 4, 6, 8)
sd_age<-4.73
ages<-c(25, 30, 35, 40)
pdp_results_post_age_2pn<-list()
for (age in ages){
  for (two_pn in two_pns){
    print(paste("Calculating partial dependence for age:", age, "and 2pn:", two_pn))
    test_modified<-post_test_cycle
    test_modified$female_age<-age/sd_age
    test_modified$n.2pn <- two_pn/sd_2pn

    mean_surv<-partial_dependence_bart(post_train_cycle, test_modified, model=post.post_cycle)
    pdp_results_post_age_2pn[[paste0("Age", age, "_2pn", two_pn)]]<-mean_surv
  }
}

df_post_age_2pn <- pdp_list_to_df(pdp_results_post_age_2pn, prefix = "")

summary(df_post_age_2pn$S)
sum(is.na(df_post_age_2pn$S))        # count non-finite S
sum(is.na(df_post_age_2pn$time))  
dim(df_post_age_2pn)
plot1<-ggplot(df_post_age_eur, aes(time, S, color = feature)) +
  geom_line(size = 1) +
  labs(title = "Embryo Utility Rate and Age",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) ) 
plot2<-ggplot(df_post_age_2pn, aes(time, S, color = feature)) +
  geom_line(size = 1) +
  labs(title = "n.2pn and Age",
       x = "Cycle", y = "S(t; x)") + theme_bw() +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5) ) 

png(filename=file.path(PLOT_PATH, "partial_dependence_plots_interaction_post.tiff"), width = 1200, height = 600)
grid.newpage()
pushViewport(viewport(layout=grid.layout(1,2)))
vplayout<-function(x,y){
  viewport(layout.pos.row=x, layout.pos.col=y)
}
print(plot1, vp=vplayout(1, 1))
print(plot2, vp=vplayout(1, 2))
dev.off()

library(dplyr)
library(stringr)

df <- df_post_age_eur %>%
  mutate(
    age = as.integer(str_match(feature, "Age(\\d+)")[,2]),
    eur = as.numeric(str_match(feature, "EUR([0-9.]+)")[,2])
  ) %>%
  mutate(
    age = factor(age, levels = sort(unique(age))),           # 25,30,35,40
    eur = factor(eur, levels = sort(unique(eur)))            # 0,0.4,0.6,0.8,1
  )
df
df2<-df_post_age_2pn %>%
  mutate(
    age = as.integer(str_match(feature, "Age(\\d+)")[,2]),
    two_pn = as.numeric(str_match(feature, "2pn([0-9.]+)")[,2])
  ) %>%
  mutate(
    age = factor(age, levels = sort(unique(age))),           # 25,30,35,40
    two_pn = factor(two_pn, levels = sort(unique(two_pn)))   # 0,2,4,6,8
  )
df2

write.csv(df, file.path(ANALYSIS_PATH, 'pdp_post_age_eur.csv'), row.names = FALSE)
write.csv(df2, file.path(ANALYSIS_PATH, 'pdp_post_age_2pn.csv'), row.names = FALSE)
df<-read.csv(file.path(ANALYSIS_PATH, 'pdp_post_age_eur.csv'))
df2<-read.csv(file.path(ANALYSIS_PATH, 'pdp_post_age_2pn.csv'))
head(df)
df$eur <- factor(df$eur)
p1 <- ggplot(df, aes(time, S, color = eur, group = eur)) +
  geom_line(linewidth = 1) +
  facet_wrap(~ age, nrow = 1) +
  scale_color_viridis_d(name = "EUR") +
  scale_x_continuous(breaks = sort(unique(df$time))) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Embryo Utilization Rate vs Cycle by Age",
       x = "Cycle", y = "S(t; x)") +
  theme_bw(base_size = 16) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
p1
df2$two_pn <- factor(df2$two_pn)
p2<-ggplot(df2, aes(time, S, color = two_pn, group = two_pn)) +
  geom_line(linewidth = 1) +
  facet_wrap(~ age, nrow = 1) +
  scale_color_viridis_d(name = "n.2pn") +
  scale_x_continuous(breaks = sort(unique(df2$time))) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "2pn zygotes vs Cycle by Age",
       x = "Cycle", y = "S(t; x)") +
  theme_bw(base_size = 16) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
p2
png(filename=file.path(PLOT_PATH, "partial_dependence_plots_age_eur_2pn.tiff"), width = 1200, height = 600)
grid.newpage()
pushViewport(viewport(layout=grid.layout(1,2)))
vplayout<-function(x,y){
  viewport(layout.pos.row=x, layout.pos.col=y)
}
print(p1, vp=vplayout(1, 1))
print(p2, vp=vplayout(1, 2))
dev.off()


