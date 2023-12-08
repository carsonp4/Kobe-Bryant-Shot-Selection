# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(stacks)
library(embed)
library(dbarts)
library(discrim)

# Reading in Data
#setwd("~/Desktop/Stat348/Kobe-Bryant-Shot-Selection/")
data <- vroom("data.csv")

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
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))

#bake(prep(my_recipe), new_data = train)

# Control settings for Stacking Models
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()
folds <- vfold_cv(train, v = 5)


# Random Forest -----------------------------------------------------------
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tg <- grid_regular(mtry(c(1,ncol(train))), min_n(), levels=7)

print("Running rf_results")
rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done rf_results")

# Boost -------------------------------------------------------------------
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

xg_tg <- dials::grid_max_entropy(xg_params, size = 30)

print("Running xg_results")
xg_results <- xg_wf %>% 
  tune_grid(resamples = folds,
            grid = xg_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done xg_results")

# BART --------------------------------------------------------------------
bart_mod <- parsnip::bart(trees = tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

bart_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(bart_mod)

bart_tg <- grid_regular(trees(), levels = 8)

print("Running bart_results")
bart_results <- bart_wf %>% 
  tune_grid(resamples = folds,
            grid = bart_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done bart_results")

# Naive Bayes -------------------------------------------------------------
nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_mod)

nb_tg <- grid_regular(Laplace(),smoothness(), levels = 10)

print("Running nb_results")
nb_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = nb_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done nb_results")


# KNN ---------------------------------------------------------------------
knn_mod <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_mod)

knn_tg <- grid_regular(neighbors(),levels = 10)

print("Running knn_results")
knn_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = knn_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done knn_results")

# SVM ---------------------------------------------------------------------
svm_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svm_mod)

svm_tg <- grid_regular(rbf_sigma(), cost(),levels = 5)

print("Running svm_results")
svm_results <- svm_wf %>% 
  tune_grid(resamples = folds,
            grid = svm_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done svm_results")

# Neural Networks ---------------------------------------------------------
nn_mod <- mlp(hidden_units = tune(),
              epochs = 500) %>% 
  set_engine("nnet") %>% 
  set_mode("classification")

nn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nn_mod)

nn_tg <- grid_regular(hidden_units(range=c(1,100)), levels=20)

print("Running nn_results")
nn_results <- nn_wf %>% 
  tune_grid(resamples = folds,
            grid = nn_tg,
            metrics = metric_set(mn_log_loss),
            control = untunedModel)

print("Done nn_results")



# Stack -------------------------------------------------------------------
# Stacking Models
the_stack <- stacks() %>% 
  add_candidates(rf_results) %>% 
  add_candidates(xg_results) %>% 
  add_candidates(bart_results) %>% 
  add_candidates(nb_results) %>% 
  add_candidates(knn_results) %>% 
  add_candidates(svm_results) %>% 
  add_candidates(nn_results) 

stack_mod <- the_stack %>% 
  blend_predictions() %>% 
  fit_members()

stack_preds <- predict(stack_mod, 
                             new_data=test,
                             type="prob")

stack_submit <- as.data.frame(cbind(test$shot_id, stack_preds$.pred_1))
colnames(stack_submit) <- c("shot_id", "shot_made_flag")
write_csv(stack_submit, "stack_submit.csv")

