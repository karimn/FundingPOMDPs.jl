library(R6)
library(magrittr)
library(tidyverse)


# Classes -----------------------------------------------------------------

source("rl_environment.R")

KBanditAction <- R6Class(
  "KBanditAction",
  inherit = Action,

  public = list(
    initialize = function(program_index, ...) {
      super$intialize(...)
      private$program_index <- program_index
    }
  ),

  active = list(
    value = function() {
      current_reward <- imap_dbl(private$current_belief$program_beliefs, 
               function(program_belief, program_index, active_program_index) program_belief$calculate_expected_reward(program_index == active_program_index),
               private$program_index) %>% 
        sum()
      
      if (private$tree_depth > 0) {
        return(current_reward + private$discount * private$simulated_updated_belief$value) 
      } else {
        return(current_reward)
      }
    },
    
    active_program_index = function() private$program
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
      private$action_list <- map(seq(k), KBanditAction$new)
    }
  ),

  active = list(
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

testenv <- create_environment(10, 5, hyperparam)
intial_action_list <- KBanditActionSet$new()
top_belief_node <- testenv$get_initial_observed_belief()
