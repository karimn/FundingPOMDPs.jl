# library(R6)
# library(magrittr)
# library(tidyverse)

# Classes -----------------------------------------------------------------

Action <- R6Class(
  "Action",

  public = list(
    initialize = function() {},
    
    expand = function(belief, discount, depth) {
      if (depth > 0) {
        return(self$calculate_reward(belief) + discount * self$calculate_expected_simulated_future_value(belief, discount, depth))
      } else {
        return(self$calculate_reward(belief))
      }
    },
    
    calculate_reward = function(belief) stop("Not implemented."), 
    calculate_expected_simulated_future_value = function(belief, discount, depth) stop("Not implemented.")
  ),

  active = list(
    evaluated_programs = function() stop("Not implemented.")
  ),

  private = list(
  )
)

ActionSet <- R6Class(
  "ActionSet",

  public = list(
    initialize = function(action_list) {
      private$action_list <- tibble(action = action_list) %>%
        mutate(value = NULL)
    },
    
    expand = function(belief, discount, depth) {
      private$action_list %<>% 
        mutate(value = map_dbl(action, ~ .x$expand(belief, discount, depth))) %>%
        # mutate(value = map_dbl(action, function(a) tryCatch({ a$expand(belief, discount, depth) }, error = function(err) browser()))) %>% 
        arrange(desc(value))
       
      return(self$value)
    },
    
    get_next_action_set = function(last_action) stop("Not implemented.") 
  ),

  active = list(
    value = function() first(private$action_list$value),
    best_action = function() first(private$action_list$action)
  ),

  private = list(
    action_list = NULL
  )
)

BeliefNode <- R6Class(
  "BeliefNode",
  
  public = list(
    
    initialize = function(current_program_beliefs, prior_belief, action_set) {
      private$current_program_beliefs <- current_program_beliefs
      private$prior_belief <- prior_belief
      private$action_set <- action_set
    
      stopifnot(all(!map_lgl(private$current_program_beliefs, is_null)))  
    },
    
    expand = function(discount, depth) {
      private$action_set$expand(self, discount, depth)
    },
    
    calculate_expected_reward = function(treated_programs) {
      imap_dbl(private$current_program_beliefs,
               function(program_belief, program_index) program_belief$calculate_expected_reward(program_index %in% treated_programs)) %>% 
        sum()
    },
    
    get_sampled_future_beliefs = function(action, ...) {
      sampled_program_beliefs <- private$current_program_beliefs %>%
        imap(function(program_belief, program_index, evaluated_programs, ...) {
          if (program_index %in% evaluated_programs) {
            program_belief$get_simulated_beliefs(...)
          } else {
            # rep(program_belief, program_belief$num_simulated_sampled)
            map(seq(program_belief$num_simulated_samples), ~ program_belief) # rep() doesn't work on env
          }
        }, evaluated_programs = action$evaluated_programs, ...) %>%
        transpose()
      
      sampled_program_beliefs %>% 
        map(BeliefNode$new, prior_belief = self, action_set = private$action_set$get_next_action_set(action))
    }
  ),
  
  active = list(
    value = function() private$action_set$value,
    program_beliefs = function() private$current_program_beliefs,
    best_action = function() private$action_set$best_action
  ),
  
  private = list(
    current_program_beliefs = NULL,
    prior_belief = NULL,
    action_set = NULL
  )
)


ProgramBelief <- R6Class(
  "ProgramBelief",
  
  public = list(
    initialize = function(dataset_draws,
                          is_simulated = TRUE,
                          prior_belief = NULL, 
                          hyperparam = if (!is_null(prior_belief)) prior_belief$hyperparam else stop("hyperparameters not specified"), 
                          num_simulated_future_datasets = 1,
                          ...) {
      private$is_simulated_belief <- is_simulated
      private$hyperparam_list <-  hyperparam
      
      private$belief_dataset <- private$get_dataset_from_draws(dataset_draws) %>% 
        list_modify(study_size = length(.$y_control))
      
      stopifnot(with(private$belief_dataset, length(y_control) == length(y_treated)))
      
      private$prior_belief <- prior_belief
      
      belief_fit <- self$fit_to_data(hyperparam, ...)
      belief_draws <- posterior::as_draws_df(belief_fit) %>% 
        ungroup()
      
      private$calculate_reward_param(belief_draws)
      private$save_simulated_future_draws(belief_draws, num_simulated_future_datasets) 
    },
    
    fit_to_data = function(hyperparam, generate_dataset_size = lst(n_control_sim = 50, n_treated_sim = 50), iter_warmup = 500, iter_sampling = iter_warmup) {
      model <- cmdstanr::cmdstan_model("sim_model.stan")
      
      model$sample(
        data = lst(
          fit = TRUE,
          sim = TRUE,
          sim_forward = TRUE,
          
          !!!generate_dataset_size,
          !!!self$all_data,
          !!!hyperparam
        ),
        iter_warmup = iter_warmup,
        iter_sampling = iter_sampling,
        chains = 4, parallel_chains = 4
      )
    },
    
    calculate_expected_reward = function(treated) {
      with(private$reward_param_mean, mu_study + treated * tau_study) 
    },
    
    get_simulated_beliefs = function(hyperparam = NULL, ...) {
      hyperparam <- hyperparam %||% self$hyperparam 
      
      private$simulated_future_draws %$%  
        map(sim_draws, ProgramBelief$new, is_simulated = TRUE, prior_belief = self, hyperparam = hyperparam, num_simulated_future_datasets = nrow(.), ...) 
    }
 ),
  
  active = list(
    hyperparam = function() private$hyperparam_list,
    num_studies = function() if (is_null(private$prior_belief)) 1 else private$prior_belief$num_studies + 1,
    num_simulated_samples = function() nrow(private$simulated_future_draws),
    is_simulated = function() private$is_simulated_belief,
    
    all_data = function() {
      if (!is_null(private$prior_belief)) {
        list_merge(private$prior_belief$all_data, !!!private$belief_dataset) %>% 
          list_modify(n_study = .$n_study + 1)
      } else {
        list_modify(private$belief_dataset, n_study = 1)
      }
    }
  ),
  
  private = list(
    hyperparam_list = NULL,
    belief_dataset = NULL,
    prior_belief = NULL,
    simulated_future_draws = NULL,
    is_simulated_belief = TRUE,
    reward_param_mean = NULL,
    
    get_dataset_from_draws = function(draws) {
      draws %>% 
        ungroup() %>% 
        select(y_control = y_control_sim, y_treated = y_treated_sim) %>% 
        as.list() 
    },
    
    calculate_reward_param = function(draws) {
      private$reward_param_mean <- draws %>% 
        tidybayes::spread_draws(mu_study[period], tau_study[period]) %>% 
        ungroup() %>% 
        filter(period == self$num_studies) %>% 
        tidybayes::mean_qi() 
      
      invisible(private$reward_param_mean)
    },
    
    save_simulated_future_draws = function(draws, num) {
      private$simulated_future_draws <- draws %>% 
        sample_n(num) %>% 
        tidybayes::spread_draws(y_control_sim[period, obs_index], y_treated_sim[period, obs_index]) %>% 
        nest(sim_draws = c(period, obs_index, matches("y_(control|treated)_sim"))) 
      
      invisible(private$simulated_future_draws)
    }
  )
)

ProgramPeriod <- R6Class(
  "ProgramPeriod",
  
  public = list(
    initialize = function(params, draws) {
      private$period_params_data <- params
      private$observed_draws <- draws 
    },
    
    get_observed_belief = function(prior_belief = NULL,
                                   hyperparam = if (!is_null(prior_belief)) prior_belief$hyperparam else stop("hyperparameters not specified"), 
                                   recalculate = FALSE,
                                   num_simulated_future_datasets = 1,
                                   ...) {
      if (is_null(private$initial_belief) || recalculate) {
        stopifnot(is_null(prior_belief) || !prior_belief$is_simulated)
        
        private$observed_belief <- ProgramBelief$new(private$observed_draws, is_simulated = FALSE, prior_belief = prior_belief, hyperparam = hyperparam, num_simulated_future_datasets = num_simulated_future_datasets, ...)
      }
      
      return(private$observed_belief)
    },
    
    get_reward = function(treated) {
      with(private$period_params_data, mu_study + treated * tau_study)
    } 
  ),
    
  active = list(
    params = function() private$period_params_data,
    dataset = function() private$period_data
  ),
    
  private = list(
    period_params_data = NULL,
    observed_draws = NULL,
    observed_belief = NULL
  )
)

Program <- R6Class(
  "Program",
  
  public = list(
    initialize = function(program_draw) {
      private$toplevel_params_data <- tidybayes::spread_draws(program_draw, `.*toplevel`, regex = TRUE)
      
      study_params <- tidybayes::spread_draws(program_draw, mu_study[period], tau_study[period], sigma_study[period])
      study_data <- tidybayes::spread_draws(program_draw, y_control_sim[period, obs_index], y_treated_sim[period, obs_index])
    
      private$period_list <- inner_join( 
        nest(study_params, params = !period),
        nest(study_data, data = !period),
        by = "period") %$% 
        map2(params, data, ProgramPeriod$new)
    }
  ),
  
  active = list(
    toplevel_params = function() private$toplevel_params_data,
    periods = function() private$period_list
  ),
  
  private = list(
    toplevel_params_data = NULL,
    period_list = NULL
  )
)

create_environment <- function(num_programs, num_periods, hyperparam, dataset_size = lst(n_control_sim = 50, n_treated_sim = 50)) {
  model <- cmdstanr::cmdstan_model("sim_model.stan")
  
  sim_fit <- model$sample(
    data = lst(
      fit = FALSE,
      sim = TRUE,
      sim_forward = FALSE,
      
      !!!dataset_size,
      
      n_study = num_periods,
      study_size = rep(0, n_study),
      
      y_control = array(dim = 0),
      y_treated = array(dim = 0),
    
      !!!hyperparam
    ),
    iter_sampling = num_programs,
    chains = 1, 
  )
  
  posterior::as_draws_df(sim_fit) %>% 
    rowwise() %>% 
    group_split() %>% 
    map(Program$new) %>% 
    Environment$new()
}

Environment <- R6Class(
  "Environment",

  public = list(
    initialize = function(programs_list) {
      private$programs_list <- programs_list 
      private$periods <- map(private$programs_list, ~ .x$periods) %>% 
        transpose()
    },
    
    get_initial_observed_belief = function(initial_action_set, hyperparam, num_simulated_future_datasets = 1) {
      private$periods %>% 
        first() %>% 
        map(~ .x$get_observed_belief(hyperparam = hyperparam, num_simulated_future_datasets = num_simulated_future_datasets)) %>% 
        BeliefNode$new(., NULL, initial_action_set)
    }
  ),
  
  active = list(
    num_programs = function() length(private$programs_list),
    num_periods = function() length(private$periods),
    programs = function() private$programs_list
  ),
  
  private = list(
    programs_list = NULL, # list(Program)
    periods = NULL, # list(ProgramPeriod)
    current_period = 1
  )
)

# Testing -----------------------------------------------------------------
# 
# hyperparam <- lst(
#   mu_sd = 2,
#   tau_mean = 0.1,
#   tau_sd = 1,
#   sigma_sd = 1,
#   eta_sd = c(1, 2, 1),
# )
# 
# test <- Environment$new(10, 5, hyperparam)
# belief <- test$programs[[1]]$periods[[1]]$get_observed_belief(hyperparam = hyperparam)
# sim_beliefs <- belief$get_simulated_beliefs(2)
# fit <- belief$fit_to_data(hyperparam, iter_warmup = 1000)
