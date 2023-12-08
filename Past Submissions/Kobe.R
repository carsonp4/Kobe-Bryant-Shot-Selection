# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)

# Reading in Data
setwd("~/Desktop/Stat348/Kobe-Bryant-Shot-Selection/")
data <- vroom("data.csv")


# EDA ---------------------------------------------------------------------

ggplot(data, aes(x = lon, y = lat)) + geom_point()


# Feature Engenieering ----------------------------------------------------
# action_type
data$action_type <- as.factor(data$action_type)

# combined_shot_type
data$combined_shot_type <- as.factor(data$combined_shot_type)

# game_event_id, game_id
data <- data %>% select(-c(game_event_id, game_id)) # not useful


# lat, loc_x, loc_y, lon
# Shot map is semi-circle so convert to polar coordinates
data$loc_r <- sqrt((data$loc_x)^2 + (data$loc_y)^2)
data$loc_theta <- atan(data$loc_y/data$loc_x)
data$loc_theta[is.na(data$loc_theta)] <- pi/2 # Remove NA's with 90 degrees
data <- data %>% select(-c(loc_x,loc_y,lat,lon))


# minutes_remaining, seconds_remaining
# Combine these two variables because useless otherwise
data$time_remaining <- (data$minutes_remaining * 60) + data$seconds_remaining
data <- data %>% select(-c(minutes_remaining, seconds_remaining))

# period, can just leave this

# playoffs, can just leave this

# season 
# Make season a factor
data$season <- as.factor(data$season)

# shot_distance, can just leave this

# shot_made_flag (Don't Change because it is response)

# shot_type, shot_zone_area, shot_zone_basic, shot_zone_range
data$shot_type <- as.factor(data$shot_type)
data$shot_zone_area <- as.factor(data$shot_zone_area)
data$shot_zone_basic <- as.factor(data$shot_zone_basic)
data$shot_zone_range <- as.factor(data$shot_zone_range)


# Shot angle instead of shot zone area ------------------------------------



# team_id, team_name
# Only ever played for Lakers
unique(data$team_id)
unique(data$team_name)
data <- data %>% select(-c(team_id, team_name))

# game_date
# Instead gong to make game number
data$game_num <- as.numeric(as.factor(data$game_date))
# Not going to do time series
data <- data %>% select(-game_date)

# matchup
# Will be redundant with home/away and opponent
data$home <- as.numeric(grepl("vs.", data$matchup, fixed = TRUE))
data$away <- as.numeric(grepl("@", data$matchup, fixed = TRUE))
data <- data %>% select(-matchup)

# opponent
data$opponent <- as.factor(data$opponent)

# shot_id, leave and will make id in recipe

# NEW: Shot in last 3 minutes of period
data$lastminutes <- ifelse(data$time_remaining <= 180, 1, 0)

###### I copied this from someone else I found online about acheivements
data$first_team <- ifelse((data$game_num >= 395 & data$game_num <= 474) | 
                             (data$game_num >= 494 & data$game_num <= 575) | 
                             (data$game_num >= 588 & data$game_num <= 651) | 
                             (data$game_num >= 740 & data$game_num <= 819) | 
                             (data$game_num >= 827 & data$game_num <= 903) | 
                             (data$game_num >= 909 & data$game_num <= 990) | 
                             (data$game_num >= 1012 & data$game_num <= 1093) | 
                             (data$game_num >= 1117 & data$game_num <= 1189) | 
                             (data$game_num >= 1213 & data$game_num <= 1294) | 
                             (data$game_num >= 1305 & data$game_num <= 1362) | 
                             (data$game_num >= 1375 & data$game_num <= 1452), 
                           1, 0)
data$scoring_leader <- ifelse((data$game_num >= 740 & data$game_num <= 819) | 
                                 (data$game_num >= 827 & data$game_num <= 903), 
                               1, 0)
data$mvp <- ifelse(data$game_num >= 909 & data$game_num <= 990, 1, 0)
data$finals_mvp <- ifelse((data$game_num >= 1112 & data$game_num <= 1116) | 
                             (data$game_num >= 1206 & data$game_num <= 1212), 
                           1, 0)
data$num_rings <- 0
data[data$game_num >= 311 & data$game_num <= 394,]$num_rings <- 1 
data[data$game_num >= 395 & data$game_num <= 493,]$num_rings <- 2 
data[data$game_num >= 494 & data$game_num <= 1116,]$num_rings <- 3 
data[data$game_num >= 1117 & data$game_num <= 1212,]$num_rings <- 4 
data[data$game_num >= 1213 & data$game_num <= 1559,]$num_rings <- 5
data$postachilles <- ifelse(data$game_num > 1452, 1, 0)


# Splitting Data ----------------------------------------------------------

train<- subset(data, !is.na(shot_made_flag))
train$shot_made_flag <- as.factor(train$shot_made_flag)
test <- subset(data, is.na(shot_made_flag)) %>% 
  select(-shot_made_flag)

my_recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  update_role(shot_id, new_role = "ID") %>% 
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))
  step_dummy(all_nominal_predictors())


bake(prep(my_recipe), new_data = train) %>% view()

# Logistic Regression -----------------------------------------------------

log_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_mod) %>%
  fit(data = train) # Fit the workflow

log_preds <- predict(log_wf,
                     new_data=test,
                     type="prob") # "class" or "prob" (see doc)

log_submit <- as.data.frame(cbind(test$shot_id, log_preds$.pred_1))
colnames(log_submit) <- c("shot_id", "shot_made_flag")
write_csv(log_submit, "log_submit.csv")

# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

#rf_tuning_grid <- grid_regular(mtry(c(1,ncol(train))), min_n(), levels=10)
rf_tuning_grid <- expand.grid(mtry = c(8,9,10), min_n = 40:45)

folds <- vfold_cv(train, v = 3, repeats = 1)


tune_control <- control_grid(verbose = TRUE)

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(mn_log_loss),
            control = tune_control)

rf_bestTune <- rf_results %>% 
  select_best("mn_log_loss")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

rf_preds <- predict(rf_final_wf,
                    new_data=test,
                    type="prob")

rf_submit <- as.data.frame(cbind(test$shot_id, rf_preds$.pred_1))
colnames(rf_submit) <- c("shot_id", "shot_made_flag")
write_csv(rf_submit, "rf_submit.csv")

# XGBoost -----------------------------------------------------------------

xg_mod <- boost_tree(trees = 1000,
                     min_n = tune(),
                     tree_depth = tune(),
                     learn_rate = tune(),
                     loss_reduction = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xg_wf <- workflow() %>% 
  add_model(xg_mod) %>% 
  add_recipe(my_recipe)

xg_params <- dials::parameters(min_n(),
                                  tree_depth(),
                                  learn_rate(),
                                  loss_reduction())

xg_grid <- dials::grid_max_entropy(xgboost_params, size = 60)

folds <- vfold_cv(train, v = 5, repeats = 1)

xg_results <- xg_wf %>% 
  tune_grid(resamples = folds,
            grid = xg_grid,
            metrics = metric_set(mn_log_loss),
            control = tune::control_grid(verbose = TRUE))

xg_bestTune <- xg_results %>% 
  select_best("mn_log_loss")

xg_final_wf <- xg_wf %>% 
  finalize_workflow(xg_bestTune) %>% 
  fit(data=train)

xg_preds <- predict(xg_final_wf,
                       new_data=test,
                       type="prob")

xg_submit <- as.data.frame(cbind(test$shot_id, xg_preds$.pred_1))
colnames(xg_submit) <- c("shot_id", "shot_made_flag")
write_csv(xg_submit, "xg_submit.csv")


#############################################################################



# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
          set_engine("ranger") %>%
          set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tuning_grid <- grid_regular(mtry(c(1,ncol(train - 1))), min_n(), levels=20)

folds <- vfold_cv(train, v = 5, repeats = 1)

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(roc_auc))

rf_bestTune <- rf_results %>% 
  select_best("roc_auc")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

rf_preds <- predict(rf_final_wf,
                    new_data=test,
                    type="prob")

rf_submit <- as.data.frame(cbind(test$id, rf_preds$.pred_1))
colnames(rf_submit) <- c("id", "ACTION")
write_csv(rf_submit, "rf_submit.csv")


# Naive Bayes -------------------------------------------------------------

nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_mod)

nb_tuning_grid <- grid_regular(Laplace(),smoothness(),
                                   levels = 5)
folds <- vfold_cv(train, v = 2, repeats=1)

nb_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc))

nb_bestTune <- nb_results %>% 
  select_best("roc_auc")

nb_final_wf <- nb_wf %>% 
  finalize_workflow(nb_bestTune) %>% 
  fit(data=train)

nb_preds <- predict(nb_final_wf,
                    new_data=test,
                    type="prob")

nb_submit <- as.data.frame(cbind(test$id, nb_preds$.pred_1))
colnames(nb_submit) <- c("id", "ACTION")
write_csv(nb_submit, "nb_submit.csv")


# KNN ---------------------------------------------------------------------
knn_mod <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")
  
knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_mod)

knn_tuning_grid <- grid_regular(neighbors(),levels = 5)
folds <- vfold_cv(train, v = 2, repeats=1)

knn_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))

knn_bestTune <- knn_results %>% 
  select_best("roc_auc")

knn_final_wf <- knn_wf %>% 
  finalize_workflow(knn_bestTune) %>% 
  fit(data=train)

knn_preds <- predict(knn_final_wf,
                    new_data=test,
                    type="prob")

knn_submit <- as.data.frame(cbind(test$id, knn_preds$.pred_1))
colnames(knn_submit) <- c("id", "ACTION")
write_csv(knn_submit, "knn_submit.csv")


# Principle Component Dimension Reduction ---------------------------------

pcrd_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .005) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9)

bake(prep(pcrd_recipe), new_data = train)


# Support Vector Machines -------------------------------------------------

svm_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svm_mod)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(),levels = 10)

folds <- vfold_cv(train, v = 5, repeats=1)

svm_results <- svm_wf %>% 
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(roc_auc))

svm_bestTune <- svm_results %>% 
  select_best("roc_auc")

svm_final_wf <- svm_wf %>% 
  finalize_workflow(svm_bestTune) %>% 
  fit(data=train)

svm_preds <- predict(svm_final_wf,
                    new_data=test,
                    type="prob")

svm_submit <- as.data.frame(cbind(test$id, svm_preds$.pred_1))
colnames(svm_submit) <- c("id", "ACTION")
write_csv(svm_submit, "svm_submit.csv")


# Balancing ---------------------------------------------------------------

smote_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_smote(all_outcomes(), neighbors=4)

smote_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

smote_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(smote_mod)

smote_tuning_grid <- grid_regular(mtry(c(1,ncol(train - 1))), min_n(), levels=8)

folds <- vfold_cv(train, v = 5, repeats = 1)

smote_results <- smote_wf %>% 
  tune_grid(resamples = folds,
            grid = smote_tuning_grid,
            metrics = metric_set(roc_auc))

smote_bestTune <- smote_results %>% 
  select_best("roc_auc")

smote_final_wf <- smote_wf %>% 
  finalize_workflow(smote_bestTune) %>% 
  fit(data=train)

smote_preds <- predict(smote_final_wf,
                    new_data=test,
                    type="prob")

smote_submit <- as.data.frame(cbind(test$id, smote_preds$.pred_1))
colnames(smote_submit) <- c("id", "ACTION")
write_csv(smote_submit, "smote_submit.csv")


# Boost -------------------------------------------------------------------
folds <- vfold_cv(train, v = 5)

#rf, smote balanced

boost_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .8) %>% 
  step_smote(all_outcomes(), neighbors=5)
  

boost_mod <- boost_tree(trees = 100,
                        min_n = tune(),
                        tree_depth = tune(),
                        learn_rate = tune(),
                        loss_reduction = tune()) %>%
  set_engine("xgboost") %>% 
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>% 
  add_model(boost_mod) 

boost_params <- dials::parameters(min_n(),
                                    tree_depth(),
                                    learn_rate(),
                                    loss_reduction())
boost_tuning_grid <- grid_max_entropy(boost_params, size = 30)

boost_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = boost_tuning_grid,
            metrics = metric_set(roc_auc))

boost_bestTune <- boost_results %>% 
  select_best("roc_auc")

boost_final_wf <- boost_wf %>% 
  finalize_workflow(boost_bestTune) %>% 
  fit(data=train)

boost_preds <- predict(boost_final_wf,
                       new_data=test,
                       type="prob")

boost_submit <- as.data.frame(cbind(test$id, boost_preds$.pred_1))
colnames(boost_submit) <- c("id", "ACTION")

#boost_submit$ACTION <- ifelse(boost_submit$ACTION > 1, 1, boost_submit$ACTION)
#boost_submit$ACTION <- ifelse(boost_submit$ACTION < 0, 0, boost_submit$ACTION)
write_csv(boost_submit, "boost_submit.csv")




# Ensemble Method ---------------------------------------------------------


