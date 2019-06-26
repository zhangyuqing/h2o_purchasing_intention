rm(list=ls())
####  Load package and initiate session
library(h2o)
h2o.init()

####  Load data into H2O
df <- h2o.importFile("online_shoppers_intention.csv")
h2o.describe(df)

# numerical features in columns 1:10
print(colnames(df)[1:10])
print(h2o.summary(df[, 1:10]))

# categorical features in columns 11:17
print(colnames(df)[11:17])
for(i in c("OperatingSystems", "Browser", "Region", "TrafficType")){
  df[, i] <- as.factor(df[, i])
}  # type conversion
h2o.describe(df)
# enum is used for categorical variables in H2O
#levels(as.data.frame(df)$Month)  # Jan and Apr not in the data

# specify target column name and pre-selected features (mutual information)
y <- "Revenue"
selected_features_MI <- c("PageValues", "ExitRates", "ProductRelated_Duration", "BounceRates", 
                          "ProductRelated", "TrafficType", "Administrative", "Month", 
                          "Administrative_Duration", "Informational")

# split into training and (balanced) test
# splits <- h2o.splitFrame(df, ratios=c(0.7, 0.2, 0.1), seed=123)
# train <- splits[[1]]
# validation <- splits[[2]]
# test <- splits[[3]]
set.seed(123)
neg_ind <- which(as.data.frame(df[, "Revenue"])=="FALSE")
pos_ind <- which(as.data.frame(df[, "Revenue"])=="TRUE")
val_test_ind <- c(sample(neg_ind, round(nrow(df)*0.15), replace=FALSE), 
                  sample(pos_ind, round(nrow(df)*0.15), replace=FALSE))
test_ind <- sort(sample(val_test_ind, round(length(val_test_ind)/3), replace=FALSE))
val_ind <- sort(setdiff(val_test_ind, test_ind))
train_ind <- setdiff(1:nrow(df), val_test_ind)

train <- df[train_ind, ]
validation <- df[val_ind, ]
test <- df[test_ind, ]

print(dim(train)); print(dim(validation)); print(dim(test))
# prevalence in train
print(table(as.data.frame(train)$Revenue)/nrow(train))
# prevalence in validation
print(table(as.data.frame(validation)$Revenue)/nrow(validation))
# prevalence in test
print(table(as.data.frame(test)$Revenue)/nrow(test))


####   Train models
# helper function to get performance
getPerformance <- function(mod, perf_val, perf_test){
  perf_metrics_val <- perf_val@metrics
  perf_metrics_test <- perf_test@metrics
  
  # accuracy
  acc_thresholds <- perf_metrics_test$thresholds_and_metric_scores$threshold
  acc_val_cutoff <- perf_metrics_val$max_criteria_and_metric_scores[
    perf_metrics_val$max_criteria_and_metric_scores$metric=="max accuracy", "threshold"]
  acc_test_cutoff <- acc_thresholds[which.min(abs(acc_thresholds - acc_val_cutoff))]
  acc_val <- perf_metrics_val$max_criteria_and_metric_scores[
    perf_metrics_val$max_criteria_and_metric_scores$metric=="max accuracy", "value"]
  acc_test <- perf_metrics_test$thresholds_and_metric_scores$f1[
    perf_metrics_test$thresholds_and_metric_scores$threshold==acc_test_cutoff]
  
  # f1
  f1_thresholds <- perf_metrics_test$thresholds_and_metric_scores$threshold
  f1_val_cutoff <- perf_metrics_val$max_criteria_and_metric_scores[
    perf_metrics_val$max_criteria_and_metric_scores$metric=="max f1", "threshold"]
  f1_test_cutoff <- f1_thresholds[which.min(abs(f1_thresholds - f1_val_cutoff))]
  f1_val <- perf_metrics_val$max_criteria_and_metric_scores[
    perf_metrics_val$max_criteria_and_metric_scores$metric=="max f1", "value"]
  f1_test <- perf_metrics_test$thresholds_and_metric_scores$f1[
    perf_metrics_test$thresholds_and_metric_scores$threshold==f1_test_cutoff]
  
  # auc
  auc_val <- h2o.auc(mod, valid=TRUE)
  auc_test <- perf_test@metrics$AUC
  
  return(c(acc_val=acc_val, acc_test=acc_test, f1_val=f1_val, f1_test=f1_test, 
           auc_val=auc_val, auc_test=auc_test))
}

## Random Forest
h2o_rf_starttime <- Sys.time()
rf <- h2o.randomForest(y=y, training_frame=train, validation_frame=validation, ntrees=100, 
                       balance_classes=TRUE, max_after_balance_size=1000)
                      #nfolds=5, fold_assignment="Stratified", 
h2o_rf_endtime <- Sys.time()
print(h2o_rf_endtime - h2o_rf_starttime)

# compute performances
rf_perf_val <- h2o.performance(rf, valid=TRUE)
rf_perf_test <- h2o.performance(rf, newdata=test)
rf_perf_summary <- getPerformance(mod=rf, perf_val=rf_perf_val, perf_test=rf_perf_test)


## DRF + feature selection
h2o.varimp_plot(rf)
print(h2o.varimp(rf))
intersect(h2o.varimp(rf)$variable[1:10], selected_features_MI)

rf_sel <- h2o.randomForest(y=y, training_frame=train, validation_frame=validation, ntrees=100,
                           balance_classes=TRUE, max_after_balance_size=1000, 
                           x=h2o.varimp(rf)$variable[1:10])
                          #nfolds=5, fold_assignment="Stratified",
                          #sample_rate_per_class=c(0.125, 1),
rf_sel_perf_val <- h2o.performance(rf_sel, valid=TRUE)
rf_sel_perf_test <- h2o.performance(rf_sel, newdata=test)
rf_sel_perf_summary <- getPerformance(mod=rf_sel, perf_val=rf_sel_perf_val, perf_test=rf_sel_perf_test)

rf_MI <- h2o.randomForest(y=y, training_frame=train, validation_frame=validation, ntrees=100,
                          balance_classes=TRUE, max_after_balance_size=1000, 
                          x=selected_features_MI)
                          #nfolds=5, fold_assignment="Stratified",
                          #sample_rate_per_class=c(0.125, 1),
rf_MI_perf_val <- h2o.performance(rf_MI, valid=TRUE)
rf_MI_perf_test <- h2o.performance(rf_MI, newdata=test)
rf_MI_perf_summary <- getPerformance(mod=rf_MI, perf_val=rf_MI_perf_val, perf_test=rf_MI_perf_test)

# unbalanced
rf_unb <- h2o.randomForest(y=y, training_frame=train, validation_frame=validation, ntrees=100)
rf_unb_perf_val <- h2o.performance(rf_unb, valid=TRUE)
rf_unb_perf_test <- h2o.performance(rf_unb, newdata=test)
rf_unb_perf_summary <- getPerformance(mod=rf_unb, perf_val=rf_unb_perf_val, perf_test=rf_unb_perf_test)

# summarize performance
h2o_rf_perf_comparison <- data.frame(
  'Model'=c("DRF", "DRF: Feature Selection (MI)", "DRF: Feature Selection (RF)", "DRF: Unbalanced"),
  'Accuracy'=round(c(rf_perf_summary["acc_test"], rf_MI_perf_summary["acc_test"], 
                     rf_sel_perf_summary["acc_test"], rf_unb_perf_summary["acc_test"]),3),
  'F1'=round(c(rf_perf_summary["f1_test"], rf_MI_perf_summary["f1_test"], 
               rf_sel_perf_summary["f1_test"], rf_unb_perf_summary["f1_test"]),3),
  'AUC'=round(c(rf_perf_summary["auc_test"], rf_MI_perf_summary["auc_test"], 
                rf_sel_perf_summary["auc_test"], rf_unb_perf_summary["auc_test"]),3)
)
print(h2o_rf_perf_comparison)


## Random Forest with ranger
library(ranger)
library(ROCR)
ranger_rf_startTime <- Sys.time()
mod_prob <- ranger::ranger(Revenue ~ ., data=as.data.frame(train), probability=TRUE,
                           num.trees=100, replace=FALSE)
ranger_rf_endTime <- Sys.time()
ranger_rf_time <- ranger_rf_endTime - ranger_rf_startTime
print(ranger_rf_time)

pred_trn_prob <- predict(mod_prob, data=as.data.frame(train))$predictions[,"TRUE"]
pred_val_prob <- predict(mod_prob, data=as.data.frame(validation))$predictions[,"TRUE"]
pred_tst_prob <- predict(mod_prob, data=as.data.frame(test))$predictions[,"TRUE"]

ranger_pred_val <- ROCR::prediction(pred_val_prob, as.data.frame(validation)$Revenue)
ranger_pred_tst <- ROCR::prediction(pred_tst_prob, as.data.frame(test)$Revenue)

ranger_auc <- as.numeric(ROCR::performance(ranger_pred_tst, "auc")@y.values)

ranger_perf_f_val <- ROCR::performance(ranger_pred_val, "f")
ranger_perf_f_test <- ROCR::performance(ranger_pred_tst, "f")
f_cutoff_val <- ranger_perf_f_val@x.values[[1]][which.max(ranger_perf_f_val@y.values[[1]])]
f_cutoff_test <- ranger_perf_f_test@x.values[[1]][which.min(abs(ranger_perf_f_test@x.values[[1]] - f_cutoff_val))]
ranger_f <- ranger_perf_f_test@y.values[[1]][ranger_perf_f_test@x.values[[1]]==f_cutoff_test]

ranger_perf_acc_val <- ROCR::performance(ranger_pred_val, "acc")
ranger_perf_acc_test <- ROCR::performance(ranger_pred_tst, "acc")
acc_cutoff_val <- ranger_perf_acc_val@x.values[[1]][which.max(ranger_perf_acc_val@y.values[[1]])]
acc_cutoff_test <- ranger_perf_acc_test@x.values[[1]][which.min(abs(ranger_perf_acc_test@x.values[[1]] - acc_cutoff_val))]
ranger_acc <- ranger_perf_acc_test@y.values[[1]][ranger_perf_acc_test@x.values[[1]]==acc_cutoff_test]



## RF: run time compare
large_train <- as.data.frame(train)[sort(sample(1:nrow(train), 10^7, replace=TRUE)), ]
rownames(large_train) <- 1:nrow(large_train)
large_train <- as.h2o(large_train)

h2o_rf_startTime <- Sys.time()
mod_h2o <- h2o.randomForest(y=y, training_frame=large_train, ntrees=100)
h2o_rf_endTime <- Sys.time()
h2o_rf_time <- h2o_rf_endTime - h2o_rf_startTime
print(h2o_rf_time)

ranger_rf_startTime <- Sys.time()
mod_ranger <- ranger::ranger(Revenue ~ ., data=as.data.frame(large_train), 
                             probability=TRUE, num.trees=100, replace=FALSE)
ranger_rf_endTime <- Sys.time()
ranger_rf_time <- ranger_rf_endTime - ranger_rf_startTime
print(ranger_rf_time)


## MLP (DL)
mlp_mod_list <- list(
  MLP_1_10 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000,  
                              model_id="MLP_1_10", hidden=10),
  MLP_1_20 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_1_20", hidden=20),
  MLP_1_40 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_1_40", hidden=40),
  MLP_2_10 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_2_10", hidden=c(10, 10)),
  MLP_2_20 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_2_20", hidden=c(20, 20)),
  MLP_2_40 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation, 
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_2_40", hidden=c(40, 40)),
  MLP_3_10 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_3_10", hidden=c(10, 10, 10)),
  MLP_3_20 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation, 
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_3_20", hidden=c(20, 20, 20)),
  MLP_3_40 = h2o.deeplearning(y=y, training_frame=train, validation_frame=validation,
                              seed=123, balance_classes=TRUE, max_after_balance_size=1000, 
                              model_id="MLP_3_40", hidden=c(40, 40, 40))
)

mlp_perfs_sub <- lapply(mlp_mod_list, function(mod){
  perf_val <- h2o.performance(mod, valid=TRUE)
  perf_test <- h2o.performance(mod, newdata=test)
  
  perf_summary <- getPerformance(mod, perf_val=perf_val, perf_test=perf_test)
  return(c(Accuracy=as.numeric(perf_summary["acc_test"]), 
           F1=as.numeric(perf_summary["f1_test"]), 
           AUC=as.numeric(perf_summary["auc_test"])))
})
names(mlp_perfs_sub) <- names(mlp_mod_list)
mlp_perfs_sub <- data.frame(Model=names(mlp_mod_list), do.call(rbind, mlp_perfs_sub))
print(mlp_perfs_sub)


## AutoML
# use original training
aml <- h2o.automl(y=y, training_frame=train, #nfolds=0, validation_frame=validation, 
                  max_runtime_secs=60, seed=123, project_name="online_shop_intention")
print(aml@leaderboard)
# prediction
aml_pred_test <- h2o.predict(aml@leader, newdata=test)  
# performance
#aml_perf_val <- h2o.performance(aml@leader, valid=TRUE)  
aml_perf_val <- h2o.performance(aml@leader, newdata=validation)
aml_perf_test <- h2o.performance(aml@leader, newdata=test)  
aml_perf_summary <- getPerformance(aml@leader, aml_perf_val, aml_perf_test)


# balance classes
aml_bl <- h2o.automl(y=y, training_frame=train, #nfolds=0, validation_frame=validation,
                     max_runtime_secs=60, seed=123, project_name="online_shop_intention_balance",
                     balance_classes=TRUE, max_after_balance_size=1000)
print(aml_bl@leaderboard, n=nrow(aml_bl@leaderboard))
# prediction
#aml_bl_pred_val <- h2o.predict(aml_bl@leader, valid=TRUE)  
aml_bl_perf_val <- h2o.performance(aml_bl@leader, newdata=validation)  
aml_bl_perf_test <- h2o.performance(aml_bl@leader, newdata=test)  
aml_bl_perf_summary <- getPerformance(aml_bl@leader, aml_bl_perf_val, aml_bl_perf_test)

aml_comparison <- data.frame('Model'=c("AutoML", "AutoML-Unbalanced"),
                             'Accuracy'=round(c(aml_bl_perf_summary['acc_test'], aml_perf_summary['acc_test']),3),
                             'F1'=round(c(aml_bl_perf_summary['f1_test'], aml_perf_summary['f1_test']),3),
                             'AUC'=round(c(aml_bl_perf_summary['auc_test'], aml_perf_summary['auc_test']),3))
print(aml_comparison)
h2o.auc(aml_perf_test)
h2o.auc(aml_bl_perf_test)

print(h2o_rf_perf_comparison)
print(mlp_perfs_sub)
print(aml_comparison)
