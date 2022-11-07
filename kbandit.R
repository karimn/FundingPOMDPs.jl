library(R6)
library(magrittr)
library(tidyverse)

# Classes -----------------------------------------------------------------

source("rl_environment.R")

KBanditAction <- R6Class(
  "KBanditAction",
  inherit = Action,

  public = list(
    initialize = function(program_index) {
      private$program_index <- program_index
    },
    
    calculate_expected_simulated_future_value = function(belief, discount, depth) { 
      belief$get_sampled_future_beliefs(self) %>%
        map_dbl(~ .x$expand(discount, depth - 1)) %>%
        mean()
    }, 
    
    calculate_reward = function(belief) {
      belief$calculate_expected_reward(private$program_index)
    }
  ),

  active = list(
    active_program_index = function() private$program_index,
    evaluated_programs = function() private$program_index
  ),

  private = list(
    program_index = NULL
  )
)

KBanditActionSet <- R6Class(
  "KBanditActionSet", 
  inherit = ActionSet,

  public = list(
    initialize = function(k) {
      map(seq(k), KBanditAction$new) %>% 
        super$initialize()
    },
    
    get_next_action_set = function(last_action) KBanditActionSet$new(self$k) 
  ),

  active = list(
    k = function() nrow(private$action_list)
  ),

  private = list(
  )
)

KBandit <- R6Class(
  "KBandit",
  
  public = list(
    initialize = function(environment) {
      private$env <- environment
    }
  ),
  
  active = list(
    k = function() private$env$num_programs
  ),
  
  private = list(
    env = NULL
  )
)

# Test --------------------------------------------------------------------

hyperparam <- lst(
  mu_sd = 2,
  tau_mean = 0.1,
  tau_sd = 1,
  sigma_sd = 1,
  eta_sd = c(1, 2, 1),
)

testenv <- create_environment(num_programs = 5, num_periods = 5, hyperparam)
initial_action_set <- KBanditActionSet$new(testenv$num_programs)
top_belief_node <- testenv$get_initial_observed_belief(initial_action_set, hyperparam, num_simulated_future_datasets = 2)
top_belief_node$expand(discount = 0.9, depth = 2)

testenv$programs %>% 
  map_dfr(~ .x$toplevel_params)

# top_belief_node$program_beliefs %>% 
#   map_dfr(~ .x$)
