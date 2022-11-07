
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
