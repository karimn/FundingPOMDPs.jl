library(R6)
library(magrittr)
library(tidyverse)

# Classes -----------------------------------------------------------------

ProgramBelief <- R6Class(
  "ProgramBelief",
  
  public = list(
    initialize = function(dataset_draws,
                          is_simulated = TRUE,
                          prior_belief = NULL, 
                          hyperparam = if (!is_null(prior_belief)) prior_belief$hyperparam else stop("hyperparameters not specified"), 
                          ...) {
      private$is_simulated_belief <- is_simulated
      private$hyperparam_list <-  hyperparam
      
      private$belief_dataset <- private$get_dataset_from_draws(dataset_draws) %>% 
        list_modify(study_size = length(.$y_control))
      
      stopifnot(with(private$belief_dataset, length(y_control) == length(y_treated)))
      
      private$prior_belief <- prior_belief
      
      beliefs_fit <- self$fit_to_data(hyperparam, ...)
      
      private$simulated_draws <- posterior::as_draws_df(beliefs_fit)
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
      reward_param_mean <- private$simulated_draws %>% 
        tidybayes::spread_draws(mu_study[period], tau_study[period]) %>% 
        ungroup() %>% 
        filter(period == self$num_studies) %>% 
        tidybayes::mean_qi() 
      
      with(reward_param_mean, mu_study + treated * tau_study) 
    },
    
    get_simulated_beliefs = function(num, hyperparam = NULL, ...) {
      hyperparam <- hyperparam %||% self$hyperparam 
      
      simulated_data <- private$simulated_draws %>% 
        tidybayes::spread_draws(y_control_sim[period, obs_index], y_treated_sim[period, obs_index]) %>% 
        nest(sim_draws = c(period, obs_index, matches("y_(control|treated)_sim"))) %>% 
        sample_n(num) %$% 
        map(sim_draws, ProgramBelief$new, is_simulated = TRUE, prior_belief = self, hyperparam = hyperparam) 
    }
 ),
  
  active = list(
    hyperparam = function() private$hyperparam_list,
    num_studies = function() if (is_null(private$prior_belief)) 1 else private$prior_belief$num_studies + 1,
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
    simulated_draws = NULL,
    is_simulated_belief = TRUE,
    
    get_dataset_from_draws = function(draws) {
      draws %>% 
        ungroup() %>% 
        select(y_control = y_control_sim, y_treated = y_treated_sim) %>% 
        as.list() 
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
    
    get_observed_belief = function(prior_belief = NULL, recalculate = FALSE, ...) {
      if (is_null(private$initial_belief) || recalculate) {
        stopifnot(is_null(prior_belief) || !prior_belief$is_simuated)
        
        private$observed_belief <- ProgramBelief$new(private$observed_draws, is_simulated = FALSE, prior_belief = prior_belief, ...)
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

Environment <- R6Class(
  "Environment",

  public = list(
    initialize = function(num_programs, num_periods, hyperparam, dataset_size = lst(n_control_sim = 50, n_treated_sim = 50)) {
      private$num_programs = num_programs
      private$num_periods = num_periods
      
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
      
      private$programs_list <- posterior::as_draws_df(sim_fit) %>% 
        rowwise() %>% 
        group_split() %>% 
        map(Program$new)
    }
  ),
  
  active = list(
    programs = function() private$programs_list
  ),
  
  private = list(
    num_programs = NULL,
    num_periods = NULL,
    programs_list = NULL
  )
)

# Testing -----------------------------------------------------------------

hyperparam <- lst(
  mu_sd = 2,
  tau_mean = 0.1,
  tau_sd = 1,
  sigma_sd = 1,
  eta_sd = c(1, 2, 1),
)

test <- Environment$new(10, 5, hyperparam)
belief <- test$programs[[1]]$periods[[1]]$get_observed_belief(hyperparam = hyperparam)
sim_beliefs <- belief$get_simulated_beliefs(2)
# fit <- belief$fit_to_data(hyperparam, iter_warmup = 1000)
