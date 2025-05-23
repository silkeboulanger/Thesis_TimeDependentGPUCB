# Load necessary libraries
library(dplyr)
library(lme4)
library(lmerTest)
library(splines)

data <- read.csv('/Users/silkedesmedt/Documents/Uni/Masterproef/Data_Fien/preprocesseddatacontrols.csv')
head(data)
data <- data %>%
  group_by(subjectID, block_nr) %>%
  filter(row_number() != 1) %>%
  ungroup() %>%
  mutate(
    # Normaliz trial and block number and reward
    trial_nr_z = as.numeric(scale(trial_nr)),
    prev_reward_z = as.numeric(scale(Previous.reward)),
    block_nr_z = scale(block_nr))

novel_click <- data$Novel.Click
high_value_click <- data$High.Value.Click
gender <- data$gender
age <- data$age
consecutive_distance <- data$log_consecutivedistance
distance_to_top10 <- data$log_distancetotop10

#Gender distribution
gender_distribution <- data %>%
  group_by(subjectID) %>%
  summarise(gender = first(gender)) %>%  # get each subject's gender once
  group_by(gender) %>%
  summarise(count = n())

print(gender_distribution)

# Age distribution
age_summary <- data %>%
  group_by(subjectID) %>%
  summarise(age = first(age)) %>%  # unique age per subject
  summarise(
    mean_age = mean(age, na.rm = TRUE),
    sd_age = sd(age, na.rm = TRUE),
    min_age = min(age, na.rm = TRUE),
    max_age = max(age, na.rm = TRUE),
    n_subjects = n()
  )

print(age_summary)

# Add gender prefer not to say to female
data$gender[data$gender == "Prefer not to say"] <- "Female"
data$gender[data$gender == "Other"] <- "Female"

## MODELS BASED ON TRIAL NUMBER ##
# Fit a mixed-effects model of consecutive distance with trial number
modelconsecutive <- lmer(consecutive_distance ~ gender + age + (trial_nr_z + block_nr_z + prev_reward_z)^2 + (1 | subjectID), data = data)
summary(modelconsecutive)

# Fit a mixed-effects model of distance to top 10 with trial number
modeldtt10 <- lmer(distance_to_top10 ~  gender + age + (trial_nr_z + block_nr_z + prev_reward_z)^2 + (1 | subjectID), data = data)
summary(modeldtt10)

# Fit a logistic mixed-effects model for the probability of novel click
modelnovel <- glmer(novel_click ~ (trial_nr_z + block_nr_z + prev_reward_z)^2 + (1 | subjectID),
               data = data,
               family = binomial)
# Summarize the model
summary(modelnovel)

# Fit a logistic mixed-effects model for the probability of highvalue click with trial number
modelhighvalue <- glmer(high_value_click ~ (trial_nr_z + block_nr_z + prev_reward_z)^2 + (1 | subjectID),
               data = data,
               family = binomial)
# Summarize the model
summary(modelhighvalue)