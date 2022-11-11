library(R6)
library(magrittr)
library(tidyverse)
library(furrr)

# Main --------------------------------------------------------------------

source("rl_environment.R")
source("rl_beliefs_and_actions.R")
source("kbandit.R")

# plan(multisession, workers = 12)

hyperparam <- lst(
  mu_sd = 2,
  tau_mean = 0.1,
  tau_sd = 1,
  sigma_sd = 1,
  eta_sd = c(1, 2, 1),
)

testenv <- create_environment(
  num_programs = 5, 
  num_periods = 3, 
  hyperparam
)

top_belief_node <- testenv$num_programs %>% 
  KBanditActionSet$new() %>% 
  testenv$solve_online_pomdp(
    hyperparam = hyperparam, 
    num_simulated_future_datasets = 1, 
    discount = 0.9, 
    plan_depth = 3 
  )

top_belief_node$print_optimal_trajectory()

testenv$programs %>% map_dfr(~ .x$toplevel_params)

plan(sequential)
