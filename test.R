library(R6)
library(magrittr)
library(tidyverse)
library(furrr)
library(profvis)

# Main --------------------------------------------------------------------

source("rl_environment.R")
source("rl_beliefs_and_actions.R")
source("kbandit.R")

plan(multisession, workers = 12)

hyperparam <- lst(
  mu_sd = 2,
  tau_mean = 0.1,
  tau_sd = 1,
  sigma_sd = 1,
  eta_sd = c(1, 2, 1),
)

testenv <- create_environment(num_programs = 5, num_periods = 5, hyperparam)
initial_action_set <- KBanditActionSet$new(testenv$num_programs)
top_belief_node <- testenv$get_initial_observed_belief(initial_action_set, hyperparam, num_simulated_future_datasets = 1)

# profvis({
  top_belief_node$expand(discount = 0.9, depth = 0)
# })

testenv$programs %>% 
  map_dfr(~ .x$toplevel_params)

# top_belief_node$program_beliefs %>% 
#   map_dfr(~ .x$)

plan(sequential)
