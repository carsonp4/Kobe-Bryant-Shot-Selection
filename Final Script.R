# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)

# Reading in Data
setwd("~/Desktop/Stat348/Kobe-Bryant-Shot-Selection/")
data <- vroom("data.csv")

# Feature Engenieering ----------------------------------------------------
### action_type, combined_shot_type
data$action_type <- as.factor(data$action_type) # Turn to factor
data$combined_shot_type <- as.factor(data$combined_shot_type) # Turn to factor

### game_event_id, game_id
data <- data %>% select(-c(game_event_id, game_id)) # not useful variables

### lat, loc_x, loc_y, lon
# Shot map is semi-circle so convert to polar coordinates
data$loc_r <- sqrt((data$loc_x)^2 + (data$loc_y)^2)
data$loc_theta <- atan(data$loc_y/data$loc_x)
data$loc_theta[is.na(data$loc_theta)] <- pi/2 # Remove NA's with 90 degrees
data <- data %>% select(-c(loc_x,loc_y,lat,lon)) # Remove old variables


### minutes_remaining, seconds_remaining
# Combine these two variables because useless otherwise
data$time_remaining <- (data$minutes_remaining * 60) + data$seconds_remaining 
data <- data %>% select(-c(minutes_remaining, seconds_remaining))# Remove old variables

### period, can just leave this since it is numeric ordered

### playoffs, can just leave this since it is binary

### season 
data$season <- as.factor(data$season) # Turn to factor

### shot_distance, can just leave this since it is numeric

### shot_made_flag (Don't Change because it is response)

### shot_type, shot_zone_area, shot_zone_basic, shot_zone_range
data$shot_type <- as.factor(data$shot_type) # Turn to factor
data$shot_zone_area <- as.factor(data$shot_zone_area) # Turn to factor
data$shot_zone_basic <- as.factor(data$shot_zone_basic) # Turn to factor
data$shot_zone_range <- as.factor(data$shot_zone_range) # Turn to factor

### team_id, team_name
# Only ever played for Lakers so we can delete these
data <- data %>% select(-c(team_id, team_name))

### game_date
# Instead going to make game number
data$game_num <- as.numeric(as.factor(data$game_date))
data <- data %>% select(-game_date) # Remove old variable

### matchup
# Will be redundant with home/away and opponent
# make a home and away variable from this data
data$home <- as.numeric(grepl("vs.", data$matchup, fixed = TRUE))
data$away <- as.numeric(grepl("@", data$matchup, fixed = TRUE))
data <- data %>% select(-matchup) # Remove old variable

### opponent
data$opponent <- as.factor(data$opponent) # Turn to factor

### shot_id, leave and will make id in recipe

### NEW: Binary if shot is in last 3 minutes of period
data$lastminutes <- ifelse(data$time_remaining <= 180, 1, 0)

### This next code is copied from Matthew Morgan
# Indicator if Kobe was all NBA first team that season
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

# Indicator if Kobe was scoring leader that season
data$scoring_leader <- ifelse((data$game_num >= 740 & data$game_num <= 819) | 
                                (data$game_num >= 827 & data$game_num <= 903), 
                              1, 0)

# Indicator if Kobe was MVP that season
data$mvp <- ifelse(data$game_num >= 909 & data$game_num <= 990, 1, 0)

# Indicator if Kobe was finals mvp that series
data$finals_mvp <- ifelse((data$game_num >= 1112 & data$game_num <= 1116) | 
                            (data$game_num >= 1206 & data$game_num <= 1212), 
                          1, 0)

# Number of rigns acquired to that points
data$num_rings <- 0
data[data$game_num >= 311 & data$game_num <= 394,]$num_rings <- 1 
data[data$game_num >= 395 & data$game_num <= 493,]$num_rings <- 2 
data[data$game_num >= 494 & data$game_num <= 1116,]$num_rings <- 3 
data[data$game_num >= 1117 & data$game_num <= 1212,]$num_rings <- 4 
data[data$game_num >= 1213 & data$game_num <= 1559,]$num_rings <- 5

# Indicator if Kobe had achilles tear yet, this affects basketball performance heavily
data$postachilles <- ifelse(data$game_num > 1452, 1, 0)


# Splitting Data ----------------------------------------------------------

train<- subset(data, !is.na(shot_made_flag)) # seperating out the training data
train$shot_made_flag <- as.factor(train$shot_made_flag) # make response a factor
test <- subset(data, is.na(shot_made_flag)) %>% # seperating out the testing data
  select(-shot_made_flag) # removing response column

my_recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  update_role(shot_id, new_role = "ID") %>% # Make shot ID not part of model
  step_dummy(all_nominal_predictors()) # dummy variables worked better than lencode_mixed


# Random Forest -----------------------------------------------------------
# Going to use a random forest model tuning mtry and min_n with the ranger engine
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Creating a workflow with recipe and model
rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

# Creating a tuning grid 
#rf_tuning_grid <- grid_regular(mtry(c(1,ncol(train))), min_n(), levels=10) #used to find first values
rf_tuning_grid <- expand.grid(mtry = c(8,9,10), min_n = 40:45) # here is the levels wanted to further analyze

# Specifing number of folds
folds <- vfold_cv(train, v = 3, repeats = 1) 

# Using tuning grid to run model with different weights
rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(mn_log_loss))

# Selecting the best model for mean log loss
rf_bestTune <- rf_results %>% 
  select_best("mn_log_loss")

# Creating a workflow with the best model
rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

# Making predicitions on the test test with the best models
rf_preds <- predict(rf_final_wf,
                    new_data=test,
                    type="prob")

# Combining prediciotns with shot id to submit
rf_submit <- as.data.frame(cbind(test$shot_id, rf_preds$.pred_1))

# Changing column names to follow format
colnames(rf_submit) <- c("shot_id", "shot_made_flag")

# Writing submission file
write_csv(rf_submit, "submission.csv")

