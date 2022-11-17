
ProgramPeriod <- R6Class(
  "ProgramPeriod",
  
  public = list(
    initialize = function(params, 
                          draws,
                          program,
                          previous_program_period = NULL) {
      private$period_params_data <- params
      private$observed_draws <- draws 
      private$program_obj <- program
      private$previous_program_period <- previous_program_period 
      private$program_period_index <- if (is_null(previous_program_period)) 1 else previous_program_period$period_index + 1
    },
    
    get_observed_belief = function(prior_belief = NULL,
                                   hyperparam = if (!is_null(prior_belief)) prior_belief$hyperparam else stop("Hyperparameters not specified."),
                                   num_simulated_future_datasets = if (!is_null(prior_belief)) prior_belief$num_simulated_samples else 1,
                                   stan_model = if (!is_null(prior_belief)) prior_belief$stan_model else stop("Stan model not specified."),
                                   ...) {
      ObservedProgramBelief$new(self, private$observed_draws, prior_belief = prior_belief, hyperparam = hyperparam, num_simulated_future_datasets = num_simulated_future_datasets, stan_model = stan_model, ...)
    },
    
    get_reward = function(treated) {
      with(private$period_params_data, mu_study + treated * tau_study)
    } 
  ),
    
  active = list(
    params = function() private$period_params_data,
    dataset = function() private$period_data,
    period_index = function() private$program_period_index,
    program = function() private$program_obj
  ),
    
  private = list(
    period_params_data = NULL,
    observed_draws = NULL,
    previous_program_period = NULL,
    program_period_index = NULL,
    program_obj = NULL
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
        accumulate2(params, data, function(prev_program, params, draws) ProgramPeriod$new(params, draws, self, if (is.R6(prev_program)) prev_program), .init = NA)[-1]
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

create_environment <- function(num_programs, num_periods, env_stan_model, env_hyperparam, dataset_size = lst(n_control_sim = 50, n_treated_sim = 50)) {
  stan_data <- lst(
    fit = FALSE,
    sim = TRUE,
    sim_forward = FALSE,
    
    !!!dataset_size,
    
    n_study = num_periods,
    study_size = rep(0, n_study),
    
    y_control = array(dim = 0),
    y_treated = array(dim = 0),
  
    !!!env_hyperparam
  )
  
  sim_fit <- env_stan_model %>%
    rstan::sampling(data = stan_data, iter = 1000 + num_programs, warmup = 1000, chains = 1)
  # sim_fit <- model$sample(data = stan_data, iter_sampling = num_programs, chains = 1)
  
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
    
    solve_online_pomdp = function(initial_action_set, discount, plan_depth, num_periods = NULL, ...) {
      stopifnot(is_null(num_periods) || num_periods <= self$num_periods)
      
      reduce(seq((num_periods %||% self$num_periods) - 1), ~ {
        cat(sprintf("[Period %d]\n", .y))
        .x$expand(discount = discount, depth = plan_depth)
        cat(sprintf("[Period %d] Executing Action %s\n", .y, .x$best_action$print_simple()))
        .x$execute_best_action(.y + 1)
      }, .init = self$get_initial_observed_belief(initial_action_set, ...))
    }, 
    
    get_initial_observed_belief = function(initial_action_set, ...) {
      private$periods %>% 
        first() %>%
        map(function(program, ...) { program$get_observed_belief(prior_belief = NULL, ...) }, ...) %>% 
        ObservedBeliefNode$new(., NULL, initial_action_set)
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
