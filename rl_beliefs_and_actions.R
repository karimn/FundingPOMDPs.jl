Action <- R6Class(
  "Action",

  public = list(
    initialize = function(last_action) {
      private$last_action <- last_action
    },
    
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
    last_action = NULL
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
        mutate(value = future_map_dbl(
        # mutate(value = map_dbl(
          action, 
          ~ .x$expand(belief, discount, depth), 
          .options = furrr_options(
            packages = c("magrittr", "tidyverse", "furrr"),
            seed = TRUE,
            globals = c("ProgramBelief", "BeliefNode", "KBanditActionSet", "ActionSet", "KBanditAction", "Action")
          )
        )) %>% 
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
      
      retry <- FALSE
      repeat {
        belief_fit <- tryCatch(
          private$fit_to_data(hyperparam, ...), 
          error = function(err) { 
            browser()
            if (retry) {
              stop(err)
            } else {
              retry <- TRUE
            }
          })
        
        if (!retry) break
      }
     
      belief_draws <- posterior::as_draws_df(belief_fit) %>% 
        ungroup()
      
      private$calculate_reward_param(belief_draws)
      private$save_simulated_future_draws(belief_draws, num_simulated_future_datasets) 
    },
    
    calculate_expected_reward = function(treated) {
      with(private$reward_param_mean, mu_study + treated * tau_study) 
    },
    
    get_simulated_beliefs = function(hyperparam = NULL, ...) {
      hyperparam <- hyperparam %||% self$hyperparam 
      
      private$simulated_future_draws %$%  
        # future_map(
        map(
          sim_draws, ProgramBelief$new, 
          is_simulated = TRUE, prior_belief = self, hyperparam = hyperparam, 
          # Here I'm only using multiple simulated datasets for the first set of simulated datasets, after that only one draw is used.
          num_simulated_future_datasets = if (self$is_simulated && private$prior_belief$is_simulated) 1 else nrow(.), ...,
          # .options = furrr_options(packages = c("magrittr", "tidyverse"), seed = TRUE)
          ) 
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
    
    fit_to_data = function(hyperparam, generate_dataset_size = lst(n_control_sim = 50, n_treated_sim = 50), iter_warmup = 500, iter_sampling = iter_warmup) {
      model <- cmdstanr::cmdstan_model("sim_model.stan")
      stan_data <- lst(
        fit = TRUE,
        sim = TRUE,
        sim_forward = TRUE,
        
        !!!generate_dataset_size,
        !!!self$all_data,
        !!!hyperparam
      )
      
      # if (self$is_simulated) {
      #   model$variational(data = stan_data)
      # } else {
      model$sample(data = stan_data, iter_warmup = iter_warmup, iter_sampling = iter_sampling, chains = 4, parallel_chains = 4, max_treedepth = 12, refresh = 0, show_messages = FALSE)
      # }
    },
    
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
        tidybayes::mean_qi(.width = c(0.8)) 
      
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