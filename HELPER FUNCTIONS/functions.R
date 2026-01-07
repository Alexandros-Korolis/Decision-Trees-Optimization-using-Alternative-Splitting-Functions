
gridsearch_one_parallel = function(labeled_train_data,
                                   target,
                                   numberoffolds = 5,
                                   choose_depth = c(3,4,5,7),
                                   choose_min_obs = c(2,3,5),
                                   choose_entr_par = 1,
                                   choose_cp = c(0,0.1,0.5),
                                   entropy_type = "Kaniadakis",
                                   metric = "recall",
                                   choose_cf = c(0.05, 0.1, 0.25, 0.4)
){
  
  # Change labeled_train_data rownames
  rownames(labeled_train_data) = seq(1,dim(labeled_train_data)[1])
  
  set.seed(123)
  # Δημιουργία 5-fold cross validation . Κάθε fold περιέχει indices.
  # Διατηρεί τα ποσοστά των τάξεων σε κάθε fold.
  cross_val_folds = createFolds(labeled_train_data[[target]], k = numberoffolds)
  
  # Hyperparameters, expand.grip -> όλοι οι δυνατοι συνδιασμοί
  parameters = expand.grid(
    depth = choose_depth,
    min_obs = choose_min_obs,
    entropy_par = choose_entr_par,
    cp = choose_cp,
    cf = choose_cf
  )
  
  # Remove parameter values for specific entropies
  if (entropy_type == "Renyi") {
    parameters = subset(parameters,entropy_par != 1)
  }
  
  
  results = foreach(i = 1:nrow(parameters), .combine = rbind,
                    .packages = c("ImbTreeEntropyKaniadakis")) %dopar% {
                      
                      parms = parameters[i,]
                      ac = c()
                      mean_ac = c()
                      sd_ac = c()
                      
                      recall_1 = c()
                      recall_2 = c()
                      recall_3 = c()
                      mean_recall = c()
                      
                      f1_1 = c()
                      f1_2 = c()
                      f1_3 = c()
                      mean_f1 = c()
                      
                      tree_leaves = c()
                      
                      for (fold in cross_val_folds) {
                        train_df = labeled_train_data[-fold,]
                        test_df = labeled_train_data[fold,]
                        
                        tree = ImbTreeEntropyKaniadakis(Y_name = target, X_names = colnames(train_df)[-ncol(train_df)], 
                                                        data = train_df, depth = parms$depth, min_obs = parms$min_obs, 
                                                        type = entropy_type, entropy_par = parms$entropy_par, 
                                                        cp = parms$cp, n_cores = 1, 
                                                        weights = NULL, cost = NULL, 
                                                        class_th = "equal", overfit = "prune", cf = parms$cf)
                        X_test = test_df[, colnames(test_df) != target, drop = FALSE]
                        predictions = PredictTree(tree,X_test)
                        matrix = confusionMatrix(predictions[[target]],test_df[[target]])
                        tree_leaves = c(tree_leaves,tree$leafCount) # leaf count
                        mean_recall = c(mean_recall, mean(matrix$byClass[,6])) # mean recall across classes
                        recall_1 = c(recall_1, matrix$byClass[1,6]) # recall class 1
                        recall_2 = c(recall_2, matrix$byClass[2,6]) # recall class 2
                        recall_3 = c(recall_3, matrix$byClass[3,6]) # recall class 3
                        mean_f1 = c(mean_f1, mean(matrix$byClass[,7])) # mean f1 across classes
                        f1_1 = c(f1_1, matrix$byClass[1,7]) # recall class 1
                        f1_2 = c(f1_2, matrix$byClass[2,7]) # recall class 2
                        f1_3 = c(f1_3, matrix$byClass[3,7]) # recall class 3
                        ac = c(ac, unname(matrix$overall[1])) # accuracy
                        
                      }
                      mean_ac = mean(ac) # mean accuracy across folds
                      sd_ac = sd(ac) # sd accuracy across folds
                      median_ac = median(ac) # median accuracy across folds
                      
                      avg_mean_recall = mean(mean_recall) # avg of mean recall across folds
                      sd_mean_recall = sd(mean_recall) # sd of mean recall across folds
                      
                      mean_recall_1 = mean(recall_1) # mean/sd recall per class across folds
                      mean_recall_2 = mean(recall_2)
                      mean_recall_3 = mean(recall_3)
                      sd_recall_1 = sd(recall_1)
                      sd_recall_2 = sd(recall_2)
                      sd_recall_3 = sd(recall_3)
                      
                      mean_f1_1 = mean(f1_1) # mean/sd f1 per class across folds
                      mean_f1_2 = mean(f1_2)
                      mean_f1_3 = mean(f1_3)
                      sd_f1_1 = sd(f1_1)
                      sd_f1_2 = sd(f1_2)
                      sd_f1_3 = sd(f1_3)
                      
                      avg_mean_f1 = mean(mean_f1) # mean/sd macro f1
                      sd_mean_f1 = sd(mean_f1)
                      
                      fitted_mean_leaves = mean(tree_leaves) # mean leaves across folds
                      fitted_sd_leaves = sd(tree_leaves) # sd leaveas across folds
                      cbind(parms, mean_ac, sd_ac, median_ac,
                            fitted_mean_leaves, fitted_sd_leaves,
                            avg_mean_recall, sd_mean_recall,
                            avg_mean_f1, sd_mean_f1,
                            mean_recall_1,sd_recall_1,
                            mean_recall_2,sd_recall_2,
                            mean_recall_3,sd_recall_3,
                            mean_f1_1, sd_f1_1,
                            mean_f1_2, sd_f1_2,
                            mean_f1_3, sd_f1_3
                      )
                    }
  
  return(list(results = results))
}

gridsearch_two_parallel = function(labeled_train_data,
                                   target,
                                   numberoffolds = 5,
                                   choose_depth = c(3,4,5),
                                   choose_min_obs = c(2,3,5),
                                   choose_entr_par_one = 1,
                                   choose_entr_par_two = 1,
                                   choose_cp = c(0,0.1,0.5),
                                   entropy_type = "Sharma-Mittal") {
  
  rownames(labeled_train_data) = seq(1, nrow(labeled_train_data))
  set.seed(123)
  cross_val_folds = createFolds(labeled_train_data[[target]], k = numberoffolds)
  
  parameters = expand.grid(
    depth = choose_depth,
    min_obs = choose_min_obs,
    entropy_par_one = choose_entr_par_one,
    entropy_par_two = choose_entr_par_two,
    cp = choose_cp
  )
  
  # Filter invalid parameter combinations for specific entropies
  if (entropy_type == "Sharma-Taneja") {
    parameters = subset(parameters, (entropy_par_one != entropy_par_two) &
                          (entropy_par_one > 0) & (entropy_par_two > 0))
  } else if (entropy_type == "Kapur") {
    parameters = subset(parameters, (entropy_par_one + entropy_par_two - 1 > 0) &
                          (entropy_par_one != 1) & (entropy_par_one > 0) & (entropy_par_two > 0))
  } else if (entropy_type == "Sharma-Mittal") {
    parameters = subset(parameters, entropy_par_one > 0)
  }
  
  results = foreach(i = 1:nrow(parameters), .combine = rbind) %dopar% {
    
    parms = parameters[i, ]
    ac = c()
    mean_recall = c()
    recall_1 = c()
    recall_2 = c()
    recall_3 = c()
    mean_f1 = c()
    f1_1 = c()
    f1_2 = c()
    f1_3 = c()
    tree_leaves = c()
    
    for (fold in cross_val_folds) {
      train_df = labeled_train_data[-fold, ]
      test_df  = labeled_train_data[fold, ]
      
      tree = ImbTreeEntropyKaniadakis(
        Y_name = target,
        X_names = colnames(train_df)[-ncol(train_df)],
        data = train_df,
        depth = parms$depth,
        min_obs = parms$min_obs,
        type = entropy_type,
        entropy_par = c(parms$entropy_par_one, parms$entropy_par_two),
        cp = parms$cp,
        n_cores = 1,
        overfit = "prune",
        cf = 0.25
      )
      
      X_test = test_df[, colnames(test_df) != target, drop = FALSE]
      predictions = PredictTree(tree, X_test)
      matrix = confusionMatrix(predictions[[target]], test_df[[target]])
      
      tree_leaves = c(tree_leaves, tree$leafCount)
      mean_recall = c(mean_recall, mean(matrix$byClass[,6]))
      recall_1 = c(recall_1, matrix$byClass[1,6])
      recall_2 = c(recall_2, matrix$byClass[2,6])
      recall_3 = c(recall_3, matrix$byClass[3,6])
      mean_f1 = c(mean_f1, mean(matrix$byClass[,7]))
      f1_1 = c(f1_1, matrix$byClass[1,7])
      f1_2 = c(f1_2, matrix$byClass[2,7])
      f1_3 = c(f1_3, matrix$byClass[3,7])
      ac = c(ac, unname(matrix$overall[1]))
    }
    
    mean_ac = mean(ac)
    sd_ac = sd(ac)
    median_ac = median(ac)
    avg_mean_recall = mean(mean_recall)
    sd_mean_recall = sd(mean_recall)
    mean_recall_1 = mean(recall_1); sd_recall_1 = sd(recall_1)
    mean_recall_2 = mean(recall_2); sd_recall_2 = sd(recall_2)
    mean_recall_3 = mean(recall_3); sd_recall_3 = sd(recall_3)
    avg_mean_f1 = mean(mean_f1); sd_mean_f1 = sd(mean_f1)
    mean_f1_1 = mean(f1_1); sd_f1_1 = sd(f1_1)
    mean_f1_2 = mean(f1_2); sd_f1_2 = sd(f1_2)
    mean_f1_3 = mean(f1_3); sd_f1_3 = sd(f1_3)
    fitted_mean_leaves = mean(tree_leaves)
    fitted_sd_leaves = sd(tree_leaves)
    
    cbind(parms, mean_ac, sd_ac, median_ac,
          fitted_mean_leaves, fitted_sd_leaves,
          avg_mean_recall, sd_mean_recall,
          avg_mean_f1, sd_mean_f1,
          mean_recall_1, sd_recall_1,
          mean_recall_2, sd_recall_2,
          mean_recall_3, sd_recall_3,
          mean_f1_1, sd_f1_1,
          mean_f1_2, sd_f1_2,
          mean_f1_3, sd_f1_3)
  }
  
  return(list(results = results))
}

one_standard_error_rule = function(sh_mit_best_parameters, entropy = "Sharma-Mittal",k=5) {
  # Selection of parameters based on mean accuracy (mean_ac)
  # Creates a threshold = max_mean - sd and keeps those with mean_ac >= threshold
  # From these, selects the parameter combination with the smallest mean_leaves
  # Supports entropies: Sharma-Mittal, Sharma-Taneja, Kapur, and single-parameter entropies
  # Standardize column names to avoid errors
  if (entropy %in% c("Sharma-Mittal", "Sharma-Taneja", "Kapur")) {
    x = sh_mit_best_parameters$results[, c("entropy_par_one", "entropy_par_two", "mean_ac", "sd_ac","fitted_mean_leaves")]
    index = which.max(x$mean_ac)
    max_mean = x$mean_ac[index]
    sd_val = x$sd_ac[index]
    threshold = max_mean - sd_val/sqrt(k)
    keep = subset(x, mean_ac >= threshold)
    best_index = which(keep$fitted_mean_leaves == min(keep$fitted_mean_leaves))
    z = keep[best_index,]
    select_par = arrange(z, fitted_mean_leaves)[1,] # in case of tie
    entropy_par_one = select_par$entropy_par_one
    entropy_par_two = select_par$entropy_par_two
    return(list(entropy_par_one = entropy_par_one, entropy_par_two = entropy_par_two))
    
  } else {
    x = sh_mit_best_parameters$results[, c("entropy_par", "mean_ac", "sd_ac","fitted_mean_leaves")]
    index = which.max(x$mean_ac)
    max_mean = x$mean_ac[index]
    sd_val = x$sd_ac[index]
    threshold = max_mean - sd_val/sqrt(k)
    keep = subset(x, mean_ac >= threshold)
    best_index = which(keep$fitted_mean_leaves == min(keep$fitted_mean_leaves))
    z = keep[best_index,]
    select_par = arrange(z, fitted_mean_leaves)[1,] # in case of tie
    entropy_par = select_par$entropy_par
    return(list(entropy_par = entropy_par))
  }
}
