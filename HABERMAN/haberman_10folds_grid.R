# Install necessary packages
install.packages("ggplot2")
library(ggplot2)

install.packages("dplyr")
library(dplyr)

install.packages("cluster")
library(cluster)

install.packages("factoextra")
library(factoextra)

install.packages("tree")
library(tree)

install.packages("rpart")
library(rpart)
library(rpart.plot)

install.packages("caret")
library(caret)

install.packages("clusterCrit")
library(clusterCrit)

install.packages("data.tree")
library(data.tree)

install.packages("plotly")
library(plotly)

library(devtools) # extention package
setwd("C:/Users/admin/ImbTreeEntropy_Kaniadakis/SOFTX-D-20-00097") # extention package
install(build = TRUE, upgrade = "never", dependencies = FALSE) # extention package
library(ImbTreeEntropyKaniadakis) # extention package

library("caret")
install.packages("DiagrammeR")
library(DiagrammeR)
library(reshape2)

install.packages("foreach") # parallel computing
library(foreach)
install.packages("doParallel") # parallel computing
library(doParallel)
install.packages("parallel") # parallel computing
library(parallel)

install.packages("rsample")
library(rsample)

install.packages("readxl")
library(readxl)

install.packages("openxlsx")   
library(openxlsx)

install.packages("rattle")
library(rattle)

install.packages("tidyr")
library(tidyr)

install.packages("tidyverse")
library(tidyverse)

install.packages("pwr")
library(pwr)

install.packages("gt")
library(gt)

# Set appropriate working directory
getwd()
setwd("C:/Users/admin/Desktop/Π.Μ.Σ Στατιστική-Χρηματοοικονομικά και Αναλογιστικά Μαθηματικά/ΜΑΘΗΜΑΤΑ 2ου Εξαμήνου/Εφαρμοσμένη Πολυμεταβλητή Ανάλυση και Big Data/Πτυχιακή")

# Import dataset 
data = read.csv("haberman.csv",sep = ";")
data$Class = data$Survival_Status
data = subset(data, select = -(Survival_Status))
data$Class = as.factor(data$Class)

# Check class balance
# 1: Patient survived 5 years or longer
# 2: Patient died within 5 years
data %>% group_by(Class) %>% count()

# power analysis
n = round(pwr.t.test(n = NULL, d = 0.6, 
                     sig.level = 0.05/21, 
                     power = 0.84, 
                     type = "paired", 
                     alternative = "two.sided")$n)

# accuracy and entropy parameters vectors
final_shannon_accuracy = c()
final_renyi_accuracy = c()
final_renyi_param = c()
final_sh_mit_accuracy = c()
final_sh_mit_param_one = c()
final_sh_mit_param_two = c()
final_tsallis_accuracy = c()
final_tsallis_param = c()
final_sh_tan_accuracy = c()
final_sh_tan_param_one = c()
final_sh_tan_param_two = c()
final_kapur_accuracy = c()
final_kapur_param_one = c()
final_kapur_param_two = c()
final_kaniad_accuracy = c()
final_kaniad_param = c()

# Height vectors
shannon_tree_height = c()
gini_height = c()
renyi_tree_height = c()
sh_mit_tree_height = c()
tsallis_tree_height = c()
sh_tan_tree_height = c()
kapur_tree_height = c()
kaniad_tree_height = c()

#Define number of leaves vectors
shannon_leaves = c()
renyi_leaves = c()
sh_mit_leaves = c()
tsallis_leaves = c()
sh_tan_leaves = c()
kapur_leaves = c()
kaniad_leaves = c()

# Recall vectors
shannon_recall_1 = c()
renyi_recall_1 = c()
sh_mit_recall_1 = c()
tsallis_recall_1 = c()
sh_tan_recall_1 = c()
kapur_recall_1 = c()
kaniad_recall_1 = c()

# F1-score vectors
shannon_f1_1 = c()
renyi_f1_1 = c()
sh_mit_f1_1 = c()
tsallis_f1_1 = c()
sh_tan_f1_1 = c()
kapur_f1_1 = c()
kaniad_f1_1 = c()


# Open cluster
cl = makeCluster(7)
registerDoParallel(cl)
start_time = Sys.time()
random_index = sample(x = 1:1000, size = n, replace = FALSE)
for (i in 1:length(random_index)) {
  set.seed(random_index[i])
  # Stratified train/test split 80-20
  split = rsample::initial_split(data, prop = 0.7, strata = Class)
  train_data = rsample::training(split)
  test_data = rsample::testing(split)
  # Remove labels from test data, save them in variables unlabeled_test_data and test_labels
  test_labels = test_data$Class # Save test dataset labels
  unlabeled_test_data = subset(x = test_data, select = -c(Class)) # Remove target variable from test data
  ################################################################################
  ############################# SHANNON ENTROPY ##################################
  ################################################################################
  
  # train/test model 
  shannon_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                          X_names = colnames(train_data)[-ncol(train_data)], 
                                          data = train_data, 
                                          depth = 50,
                                          min_obs = 2, 
                                          type = "Shannon", entropy_par = 1,
                                          cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                          class_th = "equal", overfit = "prune", cf = 0.25)
  
  shannon_predict_tree = PredictTree(shannon_tree, unlabeled_test_data) # prediction
  cm_shannon = confusionMatrix(shannon_predict_tree$Class, test_labels)
  final_shannon_accuracy = c(final_shannon_accuracy,unname(cm_shannon$overall[1]))
  shannon_tree_height = c(shannon_tree_height, shannon_tree$height-1)
  
  shannon_leaves = c(shannon_leaves, shannon_tree$leafCount) # leaves
  shannon_recall_1 = c(shannon_recall_1,cm_shannon$byClass[6]) # recall
  shannon_f1_1 = c(shannon_f1_1,cm_shannon$byClass[7]) 
  print(paste("Shannon Completed:", i))
  
  ################################################################################
  ############################# RENYI ENTROPY ####################################
  ################################################################################
  # grid search renyi 
  source("functions_binary.R")
  renyi_grid_scores = gridsearch_one_parallel(train_data,
                                              "Class",
                                              numberoffolds = 10,
                                              choose_depth = 50, 
                                              choose_min_obs = 2, 
                                              choose_entr_par = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                              choose_cp = 0,
                                              choose_cf = 0.25,
                                              entropy_type = "Renyi")
  # choose the best parameters to train the model
  best_parameter = arrange(renyi_grid_scores$results, desc(mean_ac),sd_ac,fitted_mean_leaves,fitted_sd_leaves,desc(avg_mean_recall))[1,]
  
  
  # train/test model 
  renyi_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                        X_names = colnames(train_data)[-ncol(train_data)], 
                                        data = train_data, depth = 50
                                        , min_obs = 2, 
                                        type = "Renyi", entropy_par = best_parameter$entropy_par,
                                        cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                        class_th = "equal", overfit = "prune", cf = 0.25)
  
  renyi_predict_tree = PredictTree(renyi_tree, unlabeled_test_data) # prediction
  cm_renyi = confusionMatrix(renyi_predict_tree$Class, test_labels)
  final_renyi_accuracy = c(final_renyi_accuracy,unname(cm_renyi$overall[1]))
  final_renyi_param = c(final_renyi_param,best_parameter$entropy_par)
  renyi_tree_height = c(renyi_tree_height, renyi_tree$height-1)
  renyi_leaves = c(renyi_leaves,renyi_tree$leafCount)
  renyi_recall_1 = c(renyi_recall_1,cm_renyi$byClass[6]) # recall
  renyi_f1_1 = c(renyi_f1_1,cm_renyi$byClass[7]) # f1
  print(paste("Renyi Completed:", i))
  ################################################################################
  ######################### SHARMA - MITTAL ENTROPY ##############################
  ################################################################################
  # grid search  
  source("functions_binary.R")
  sh_mit_grid_scores = gridsearch_two_parallel(train_data,
                                               "Class",
                                               numberoffolds = 10,
                                               choose_depth = 50,
                                               choose_min_obs = 2, # default
                                               choose_entr_par_one = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                               choose_entr_par_two = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                               choose_cp = 0,
                                               entropy_type = "Sharma-Mittal")
  # choose the best parameters to train the model
  best_parameter = arrange(sh_mit_grid_scores$results, desc(mean_ac),sd_ac,fitted_mean_leaves,fitted_sd_leaves,desc(avg_mean_recall))[1,]
  
  
  # train/test model 
  sh_mit_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                         X_names = colnames(train_data)[-ncol(train_data)], 
                                         data = train_data, depth = 50
                                         , min_obs = 2, 
                                         type = "Sharma-Mittal", 
                                         entropy_par = c(best_parameter$entropy_par_one,best_parameter$entropy_par_two),
                                         cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                         class_th = "equal", overfit = "prune", cf = 0.25)
  
  sh_mit_predict_tree = PredictTree(sh_mit_tree, unlabeled_test_data) # prediction
  cm_sh_mit = confusionMatrix(sh_mit_predict_tree$Class, test_labels)
  final_sh_mit_accuracy = c(final_sh_mit_accuracy,unname(cm_sh_mit$overall[1]))
  final_sh_mit_param_one = c(final_sh_mit_param_one,best_parameter$entropy_par_one)
  final_sh_mit_param_two = c(final_sh_mit_param_two,best_parameter$entropy_par_two)
  sh_mit_tree_height = c(sh_mit_tree_height, sh_mit_tree$height-1)
  sh_mit_leaves = c(sh_mit_leaves,sh_mit_tree$leafCount)
  sh_mit_recall_1 = c(sh_mit_recall_1,cm_sh_mit$byClass[6]) # recall
  sh_mit_f1_1 = c(sh_mit_f1_1,cm_sh_mit$byClass[7]) # f1
  print(paste("Sharma Mittal Completed:", i))
  ################################################################################
  ######################### TSALLIS ENTROPY ######################################
  ################################################################################
  
  # grid search 
  source("functions_binary.R")
  tsallis_grid_scores = gridsearch_one_parallel(train_data,
                                                "Class",
                                                numberoffolds = 10,
                                                choose_depth = 50,
                                                choose_min_obs = 5,
                                                choose_entr_par = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                                choose_cp = 0,
                                                choose_cf = 0.25,
                                                entropy_type = "Tsallis")
  
  # choose the best parameters to train the model
  best_parameter = arrange(tsallis_grid_scores$results, desc(mean_ac),sd_ac,fitted_mean_leaves,fitted_sd_leaves,desc(avg_mean_recall))[1,]
  
  
  # train/test model 
  tsallis_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                          X_names = colnames(train_data)[-ncol(train_data)], 
                                          data = train_data, depth = 50
                                          , min_obs = 2, 
                                          type = "Tsallis", entropy_par = best_parameter$entropy_par,
                                          cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                          class_th = "equal", overfit = "prune", cf = 0.25)
  
  tsallis_predict_tree = PredictTree(tsallis_tree, unlabeled_test_data) # prediction
  cm_tsallis = confusionMatrix(tsallis_predict_tree$Class, test_labels)
  final_tsallis_accuracy = c(final_tsallis_accuracy,unname(cm_tsallis$overall[1]))
  final_tsallis_param = c(final_tsallis_param,best_parameter$entropy_par)
  tsallis_tree_height = c(tsallis_tree_height, tsallis_tree$height-1)
  tsallis_leaves = c(tsallis_leaves,tsallis_tree$leafCount)
  tsallis_recall_1 = c(tsallis_recall_1,cm_tsallis$byClass[6]) # recall
  tsallis_f1_1 = c(tsallis_f1_1,cm_tsallis$byClass[7]) # f1
  print(paste("Tsallis Completed:", i))
  ################################################################################
  ######################### SHARMA-TANEJA ENTROPY ################################
  ################################################################################
  # grid search 
  source("functions_binary.R")
  st_best_grid_scores = gridsearch_two_parallel(train_data,
                                                "Class",
                                                numberoffolds = 10,
                                                choose_depth = 50,
                                                choose_min_obs = 2,
                                                choose_entr_par_one = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                                choose_entr_par_two = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                                choose_cp = 0,
                                                entropy_type = "Sharma-Taneja")
  # choose the best parameters to train the model
  best_parameter = arrange(st_best_grid_scores$results, desc(mean_ac),sd_ac,fitted_mean_leaves,fitted_sd_leaves,desc(avg_mean_recall))[1,]
  
  
  # train/test model 
  sh_tan_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                         X_names = colnames(train_data)[-ncol(train_data)], 
                                         data = train_data, depth = 50
                                         , min_obs = 2, 
                                         type = "Sharma-Taneja", 
                                         entropy_par = c(best_parameter$entropy_par_one,best_parameter$entropy_par_two),
                                         cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                         class_th = "equal", overfit = "prune", cf = 0.25)
  
  sh_tan_predict_tree = PredictTree(sh_tan_tree, unlabeled_test_data) # prediction
  cm_sh_tan = confusionMatrix(sh_tan_predict_tree$Class, test_labels)
  final_sh_tan_accuracy = c(final_sh_tan_accuracy,unname(cm_sh_tan$overall[1]))
  final_sh_tan_param_one = c(final_sh_tan_param_one,best_parameter$entropy_par_one)
  final_sh_tan_param_two = c(final_sh_tan_param_two,best_parameter$entropy_par_two)
  sh_tan_tree_height = c(sh_tan_tree_height, sh_tan_tree$height-1)
  sh_tan_leaves = c(sh_tan_leaves,sh_tan_tree$leafCount)
  sh_tan_recall_1 = c(sh_tan_recall_1,cm_sh_tan$byClass[6]) # recall
  sh_tan_f1_1 = c(sh_tan_f1_1,cm_sh_tan$byClass[7]) # f1
  print(paste("Sharma Taneja Completed:", i))
  ################################################################################
  ######################### KAPUR ENTROPY ########################################
  ################################################################################
  # grid search
  source("functions_binary.R")
  kapur_grid_scores = gridsearch_two_parallel(train_data,
                                              "Class",
                                              numberoffolds = 10,
                                              choose_depth = 50,
                                              choose_min_obs = 2,
                                              choose_entr_par_one = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                              choose_entr_par_two = c(0.25,0.5,0.75,1.25,1.5,2,2.25,2.5,3,3.25,3.5,3.75,4),
                                              choose_cp = 0,
                                              entropy_type = "Kapur")
  # choose the best parameters to train the model
  best_parameter = arrange(kapur_grid_scores$results, desc(mean_ac),sd_ac,fitted_mean_leaves,fitted_sd_leaves,desc(avg_mean_recall))[1,]
  
  
  # train/test model 
  kapur_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                        X_names = colnames(train_data)[-ncol(train_data)], 
                                        data = train_data, depth = 50
                                        , min_obs = 2, 
                                        type = "Kapur", 
                                        entropy_par = c(best_parameter$entropy_par_one,best_parameter$entropy_par_two),
                                        cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                        class_th = "equal", overfit = "prune", cf = 0.25)
  
  kapur_predict_tree = PredictTree(kapur_tree, unlabeled_test_data) # prediction
  cm_kapur = confusionMatrix(kapur_predict_tree$Class, test_labels)
  final_kapur_accuracy = c(final_kapur_accuracy,unname(cm_kapur$overall[1]))
  final_kapur_param_one = c(final_kapur_param_one,best_parameter$entropy_par_one)
  final_kapur_param_two = c(final_kapur_param_two,best_parameter$entropy_par_two)
  kapur_tree_height = c(kapur_tree_height, kapur_tree$height-1)
  kapur_leaves = c(kapur_leaves,kapur_tree$leafCount)
  kapur_recall_1 = c(kapur_recall_1,cm_kapur$byClass[6]) # recall
  kapur_f1_1 = c(kapur_f1_1,cm_kapur$byClass[7]) # f1
  print(paste("Kapur Completed:", i))
  ################################################################################
  ######################### KANIADAKIS ENTROPY ###################################
  ################################################################################
  # grid search 
  source("functions_binary.R")
  kaniad_grid_scores = gridsearch_one_parallel(train_data,
                                               "Class",
                                               numberoffolds = 10,
                                               choose_depth = 50,
                                               choose_min_obs = 2,
                                               choose_entr_par = c(seq(0.1,0.9,0.1)),
                                               choose_cp = 0,
                                               choose_cf = 0.25,
                                               entropy_type = "Kaniadakis")
  # choose the best parameters to train the model
  best_parameter = arrange(kaniad_grid_scores$results, desc(mean_ac),sd_ac,fitted_mean_leaves,fitted_sd_leaves,desc(avg_mean_recall))[1,]
  
  # train/test model 
  kaniad_tree = ImbTreeEntropyKaniadakis(Y_name = "Class", 
                                         X_names = colnames(train_data)[-ncol(train_data)], 
                                         data = train_data, depth = 50
                                         , min_obs = 2, 
                                         type = "Kaniadakis", 
                                         entropy_par = best_parameter$entropy_par,
                                         cp = 0, n_cores = 1, weights = NULL, cost = NULL, 
                                         class_th = "equal", overfit = "prune", cf = 0.25)
  
  kaniad_predict_tree = PredictTree(kaniad_tree, unlabeled_test_data) # prediction
  cm_kaniad = confusionMatrix(kaniad_predict_tree$Class, test_labels)
  final_kaniad_accuracy = c(final_kaniad_accuracy,unname(cm_kaniad$overall[1]))
  final_kaniad_param = c(final_kaniad_param,best_parameter$entropy_par)
  kaniad_tree_height = c(kaniad_tree_height, kaniad_tree$height-1)
  kaniad_leaves = c(kaniad_leaves,kaniad_tree$leafCount)
  kaniad_recall_1 = c(kaniad_recall_1,cm_kaniad$byClass[6]) # recall
  kaniad_f1_1 = c(kaniad_f1_1,cm_kaniad$byClass[7]) # f1
  print(paste("Kaniad Completed:", i))
  print(paste("----------Completed Round:",i))
}

# close cluster 
stopCluster(cl)

end_time = Sys.time()  
elapsed = end_time - start_time
print(elapsed)

results_shannon = data.frame(
  accuracy = final_shannon_accuracy,
  depth = shannon_tree_height,
  leaves = shannon_leaves,
  recall_1 = shannon_recall_1,
  f1_1 = shannon_f1_1
)

results_renyi = data.frame(
  accuracy = final_renyi_accuracy,
  entropy_parameter = final_renyi_param,
  depth = renyi_tree_height,
  leaves = renyi_leaves,
  recall_1 = renyi_recall_1,
  f1_1 = renyi_f1_1
)

results_sh_mit = data.frame(
  accuracy = final_sh_mit_accuracy,
  entropy_parameter_one = final_sh_mit_param_one,
  entropy_parameter_two = final_sh_mit_param_two,
  depth = sh_mit_tree_height,
  leaves = sh_mit_leaves,
  recall_1 = sh_mit_recall_1,
  f1_1 = sh_mit_f1_1
)

results_tsallis = data.frame(
  accuracy = final_tsallis_accuracy,
  entropy_parameter = final_tsallis_param,
  depth = tsallis_tree_height,
  leaves = tsallis_leaves,
  recall_1 = tsallis_recall_1,
  f1_1 = tsallis_f1_1
)

results_sh_tan = data.frame(
  accuracy = final_sh_tan_accuracy,
  entropy_parameter_one = final_sh_tan_param_one,
  entropy_parameter_two = final_sh_tan_param_two,
  depth = sh_tan_tree_height,
  leaves = sh_tan_leaves,
  recall_1 = sh_tan_recall_1,
  f1_1 = sh_tan_f1_1
)

results_kapur = data.frame(
  accuracy = final_kapur_accuracy,
  entropy_parameter_one = final_kapur_param_one,
  entropy_parameter_two = final_kapur_param_two,
  depth = kapur_tree_height,
  leaves = kapur_leaves,
  recall_1 = kapur_recall_1,
  f1_1 = kapur_f1_1
)

results_kaniad = data.frame(
  accuracy = final_kaniad_accuracy,
  entropy_parameter = final_kaniad_param,
  depth = kaniad_tree_height,
  leaves = kaniad_leaves,
  recall_1 = kaniad_recall_1,
  f1_1 = kaniad_f1_1
)

index = data.frame(seed = random_index)


install.packages("openxlsx")
library(openxlsx)
work_book = createWorkbook()
addWorksheet(work_book, "Shannon")
writeData(work_book, sheet = "Shannon", results_shannon)
addWorksheet(work_book, "Renyi")
writeData(work_book, sheet = "Renyi", results_renyi)
addWorksheet(work_book, "Sharma_Mittal")
writeData(work_book, sheet = "Sharma_Mittal", results_sh_mit)
addWorksheet(work_book, "Tsallis")
writeData(work_book, sheet = "Tsallis", results_tsallis)
addWorksheet(work_book, "Sharma_Taneja")
writeData(work_book, sheet = "Sharma_Taneja", results_sh_tan)
addWorksheet(work_book, "Kapur")
writeData(work_book, sheet = "Kapur", results_kapur)
addWorksheet(work_book, "Kaniadakis")
writeData(work_book, sheet = "Kaniadakis", results_kaniad)
addWorksheet(work_book, "index")
writeData(work_book, sheet = "index", index)
saveWorkbook(work_book, "repeated_holdout_cross_val_51_haberman.xlsx", overwrite = TRUE)


################################################################################ accuracy
# load results data 
results_shannon = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                            sheet = "Shannon")

results_renyi = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                          sheet = "Renyi")

results_sh_mit = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                           sheet = "Sharma_Mittal")

results_sh_tan = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                           sheet = "Sharma_Taneja")

results_tsallis = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                            sheet = "Tsallis")

results_kapur = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                          sheet = "Kapur")

results_kaniad = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                           sheet = "Kaniadakis")

index = read.xlsx("repeated_holdout_cross_val_50_haberman.xlsx",
                  sheet = "index")
# Vertical side by side box plots for accuracy 
df_accuracy = data.frame(results_shannon$accuracy,results_renyi$accuracy,results_sh_mit$accuracy,
                         results_tsallis$accuracy,results_sh_tan$accuracy,results_kapur$accuracy,
                         results_kaniad$accuracy)
colnames(df_accuracy) = c("Shannon","Renyi","Sharma-Mittal","Tsallis","Sharma-Taneja","Kapur","Kaniadakis")
df_accuracy_long = df_accuracy %>% pivot_longer(cols = everything(),names_to = "Εντροπία", values_to = "Ακρίβεια")
ggplot(data = df_accuracy_long, mapping = aes(x = Εντροπία,y = Ακρίβεια, fill = Εντροπία))+
  geom_boxplot()+theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))+ggtitle("Ακρίβεια")

summary(results_shannon)
summary(results_renyi)
summary(results_sh_mit)
summary(results_tsallis)
summary(results_sh_tan)
summary(results_kapur)
summary(results_kaniad)

# Paired t-tests
num_test = 21

t.test(results_kapur$accuracy,results_shannon$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_kapur$accuracy - results_shannon$accuracy)/sd(results_kapur$accuracy - results_shannon$accuracy)

t.test(results_kaniad$accuracy,results_shannon$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_kaniad$accuracy - results_shannon$accuracy)/sd(results_kaniad$accuracy - results_shannon$accuracy)

t.test(results_renyi$accuracy,results_shannon$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_renyi$accuracy - results_shannon$accuracy)/sd(results_renyi$accuracy - results_shannon$accuracy)

t.test(results_sh_mit$accuracy,results_shannon$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_sh_mit$accuracy - results_shannon$accuracy)/sd(results_sh_mit$accuracy - results_shannon$accuracy)

t.test(results_sh_tan$accuracy,results_shannon$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_sh_tan$accuracy - results_shannon$accuracy)/sd(results_sh_tan$accuracy - results_shannon$accuracy)

t.test(results_tsallis$accuracy,results_shannon$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_tsallis$accuracy - results_shannon$accuracy)/sd(results_tsallis$accuracy - results_shannon$accuracy)


t.test(results_kaniad$accuracy,results_sh_mit$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_kaniad$accuracy - results_sh_mit$accuracy)/sd(results_kaniad$accuracy - results_sh_mit$accuracy)

t.test(results_kaniad$accuracy,results_sh_tan$accuracy,paired = TRUE)$p.value< (0.05/num_test)
d = mean(results_kaniad$accuracy - results_sh_tan$accuracy)/sd(results_kaniad$accuracy - results_sh_tan$accuracy)

t.test(results_kaniad$accuracy,results_tsallis$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_kaniad$accuracy - results_tsallis$accuracy)/sd(results_kaniad$accuracy - results_tsallis$accuracy)

t.test(results_kaniad$accuracy,results_kapur$accuracy,paired = TRUE)$p.value< (0.05/num_test)
d = mean(results_kaniad$accuracy - results_kapur$accuracy)/sd(results_kaniad$accuracy - results_kapur$accuracy)

t.test(results_kaniad$accuracy,results_renyi$accuracy,paired = TRUE)$p.value< (0.05/num_test)
d = mean(results_kaniad$accuracy - results_renyi$accuracy)/sd(results_kaniad$accuracy - results_renyi$accuracy)

t.test(results_sh_mit$accuracy,results_sh_tan$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
t.test(results_sh_mit$accuracy,results_tsallis$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_sh_mit$accuracy - results_tsallis$accuracy)/sd(results_sh_mit$accuracy - results_tsallis$accuracy)
t.test(results_sh_mit$accuracy,results_kapur$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
t.test(results_sh_mit$accuracy,results_renyi$accuracy,paired = TRUE)$p.value< (0.05/num_test) 

t.test(results_kapur$accuracy,results_renyi$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
t.test(results_kapur$accuracy,results_sh_tan$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
t.test(results_kapur$accuracy,results_tsallis$accuracy,paired = TRUE)$p.value< (0.05/num_test) 

t.test(results_renyi$accuracy,results_sh_tan$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
t.test(results_renyi$accuracy,results_tsallis$accuracy,paired = TRUE)$p.value< (0.05/num_test) 
d = mean(results_renyi$accuracy - results_tsallis$accuracy)/sd(results_renyi$accuracy - results_tsallis$accuracy)
t.test(results_sh_tan$accuracy,results_tsallis$accuracy,paired = TRUE)$p.value< (0.05/num_test) 

# Error bars 
entropy = data.frame(Measure = c("Shannon", "Renyi", "Sharma-Mittal","Tsallis", "Sharma-Taneja", "Kapur", "Kaniadakis"),
                     Mean = round(c(mean(results_shannon$accuracy), mean(results_renyi$accuracy), mean(results_sh_mit$accuracy),
                                    mean(results_tsallis$accuracy), mean(results_sh_tan$accuracy), mean(results_kapur$accuracy), 
                                    mean(results_kaniad$accuracy)),3),
                     SD   = round(c(sd(results_shannon$accuracy), sd(results_renyi$accuracy), sd(results_sh_mit$accuracy),
                                    sd(results_tsallis$accuracy), sd(results_sh_tan$accuracy), sd(results_kapur$accuracy), 
                                    sd(results_kaniad$accuracy)),3))

ggplot(entropy, aes(x = Measure, y = Mean)) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean - SD, ymax = Mean + SD),
    width = 0.2
  ) +
  ylab("Ακρίβεια") +
  xlab("Εντροπία") +
  ggtitle("Μέση Τιμή Ακρίβειας ± SD Ακρίβειας") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

colnames(entropy) = c("Εντροπία", "Μέση Τιμή", "Τυπική Απόκλιση")
entropy %>% gt()

# Complexity (depth)
entropy_depth = data.frame(Measure = c("Shannon", "Renyi", "Sharma-Mittal","Tsallis", "Sharma-Taneja", "Kapur", "Kaniadakis"),
                           Min = c(min(results_shannon$depth), min(results_renyi$depth), min(results_sh_mit$depth),
                                   min(results_tsallis$depth), min(results_sh_tan$depth), min(results_kapur$depth), 
                                   min(results_kaniad$depth)),
                           Max = c(max(results_shannon$depth), max(results_renyi$depth), max(results_sh_mit$depth),
                                   max(results_tsallis$depth), max(results_sh_tan$depth), max(results_kapur$depth), 
                                   max(results_kaniad$depth)),
                           Mean = round(c(mean(results_shannon$depth), mean(results_renyi$depth), mean(results_sh_mit$depth),
                                          mean(results_tsallis$depth), mean(results_sh_tan$depth), mean(results_kapur$depth), 
                                          mean(results_kaniad$depth)),3),
                           SD   = round(c(sd(results_shannon$depth), sd(results_renyi$depth), sd(results_sh_mit$depth),
                                          sd(results_tsallis$depth), sd(results_sh_tan$depth), sd(results_kapur$depth), 
                                          sd(results_kaniad$depth)),3))
colnames(entropy_depth) = c("Εντροπία", "Ελάχιστη Τιμή","Μέγιστη Τιμή","Μέση Τιμή", "Τυπική Απόκλιση")

entropy_depth %>% gt()

# Complexity (leaves)
entropy_leaves = data.frame(Measure = c("Shannon", "Renyi", "Sharma-Mittal","Tsallis", "Sharma-Taneja", "Kapur", "Kaniadakis"),
                            Min = c(min(results_shannon$leaves), min(results_renyi$leaves), min(results_sh_mit$leaves),
                                    min(results_tsallis$leaves), min(results_sh_tan$leaves), min(results_kapur$leaves), 
                                    min(results_kaniad$leaves)),
                            Max = c(max(results_shannon$leaves), max(results_renyi$leaves), max(results_sh_mit$leaves),
                                    max(results_tsallis$leaves), max(results_sh_tan$leaves), max(results_kapur$leaves), 
                                    max(results_kaniad$leaves)),
                            Mean = round(c(mean(results_shannon$leaves), mean(results_renyi$leaves), mean(results_sh_mit$leaves),
                                           mean(results_tsallis$leaves), mean(results_sh_tan$leaves), mean(results_kapur$leaves), 
                                           mean(results_kaniad$leaves)),3),
                            SD   = round(c(sd(results_shannon$leaves), sd(results_renyi$leaves), sd(results_sh_mit$leaves),
                                           sd(results_tsallis$leaves), sd(results_sh_tan$leaves), sd(results_kapur$leaves), 
                                           sd(results_kaniad$leaves)),3))
colnames(entropy_leaves) = c("Εντροπία", "Ελάχιστη Τιμή","Μέγιστη Τιμή","Μέση Τιμή", "Τυπική Απόκλιση")

entropy_leaves %>% gt()
