Action <- R6Class(
  "Action",

  public = list(
    initialize = function(implemented_program_indices, action_set, evaluated_program_indices = implemented_program_indices, last_action = NULL) {
      private$implemented_program_indices <- implemented_program_indices
      private$evaluated_program_indices <- evaluated_program_indices
      private$last_action <- last_action
      private$parent_action_set <- action_set
    }, 
    
    expand = function(belief, discount, depth) {
      private$action_value <- if (is_null(private$action_value)) {
        if (depth > 0) {
          return(self$calculate_reward(belief) + discount * self$calculate_expected_simulated_future_value(belief, discount, depth))
        } else {
          self$calculate_reward_range(belief)["lower"]
        }
      } else private$action_value
      
      invisible(private$action_value)
    },
    
    execute = function(belief, current_period, ...) {
      stopifnot(!any(map_lgl(belief$program_beliefs, ~ .x$is_simulated)))
      
      updated_program_beliefs <- belief$program_beliefs
      
      if (!is_null(private$evaluated_program_indices)) {
        updated_program_beliefs[private$evaluated_program_indices] %<>% 
          map(function(program_belief, ...) program_belief$get_next_observed_belief(current_period, ...), ...)
      }
      
      ObservedBeliefNode$new(updated_program_beliefs, belief, private$parent_action_set$get_next_action_set(self)) 
    },
    
    calculate_reward = function(belief) belief$calculate_expected_reward(private$implemented_program_indices),
    calculate_true_reward = function(observed_belief) observed_belief$calculate_true_reward(private$implemented_program_indices),
    calculate_reward_range = function(belief, width = 0.8) belief$calculate_expected_reward_range(private$implemented_program_indices, width),
    
    calculate_expected_simulated_future_value = function(belief, discount, depth) { 
      belief$get_sampled_future_beliefs(self) %>%
        # map_dbl(
        future_map_dbl(
          ~ .x$expand(discount, depth - 1),
          .options = furrr_options(
            packages = c("magrittr", "tidyverse", "furrr"),
            seed = TRUE,
            globals = c("ProgramBelief", "BeliefNode", "KBanditActionSet", "ActionSet", "Action")
          ),
          .progress = FALSE 
        ) %>%
        mean()
    }, 
    
    calculate_state_rewards = function(belief) belief$calculate_state_rewards(private$implemented_program_indices), 
    
    print_simple = function(belief = NULL) {
      action_desc <- sprintf("[Impl: {%s}, Eval: {%s}]", str_c(private$implemented_program_indices, collapse = ", "), str_c(private$evaluated_program_indices, collapse = ", "))
      
      if (!is_null(belief)) {
        sprintf("(%s, %.2f)", action_desc, self$calculate_reward(belief))
      } else {
        action_desc
      } 
    } 
  ),

  active = list(
    implemented_programs = function() private$implemented_program_indices,
    evaluated_programs = function() private$evaluated_program_indices,
    action_set = function() private$parent_action_set,
    value = function() private$action_value
  ),
  
  private = list(
    implemented_program_indices = NULL,
    evaluated_program_indices = NULL,
    last_action = NULL,
    parent_action_set = NULL,
    action_value = NULL
  )
)

ActionSet <- R6Class(
  "ActionSet",

  public = list(
    initialize = function(action_list, last_action = NULL) {
      private$action_list <- tibble(action = action_list) %>%
        mutate(value = -Inf)
    },
    
    expand = function(belief, discount, depth) {
      private$action_list %<>% 
        mutate(map_dfr(action, ~ { .x$calculate_reward_range(belief) })) %>%
        mutate(
          old_upper = upper + max(upper) * discount * (1 - discount^depth) / (1 - discount),
          
          # FIB offline upper-bound calculation
          # upper = map_dbl(action, ~ .x$calculate_reward(belief)) + (discount / (1 - discount)) * self$calculate_mean_max_reward(belief)  
          upper = map_dbl(action, ~ .x$calculate_reward(belief)) + (discount * (1 - discount^depth) / (1 - discount)) * self$calculate_mean_max_reward(belief)  
        ) %>%
        arrange(desc(upper))
      
      # Real-Time Belief Space Search
      
      if (depth > 0) {
        num_actions <- nrow(private$action_list) 
        action_index <- 1
        highest_lb <- -Inf
       
        while(action_index <= num_actions && private$action_list$upper[action_index] >= highest_lb)  {
          curr_action <- private$action_list$action[[action_index]]
          
          cat(sprintf("[Depth = %d] Expanding (%d) %s\n", depth, action_index, curr_action$print_simple()))
          
          current_value <- curr_action$expand(belief, discount, depth)
          private$action_list$value[action_index] <- current_value
          highest_lb <- max(highest_lb, current_value)
          action_index <- action_index + 1
        }
       
        num_pruned_actions <- num_actions - action_index + 1
        
        if (num_pruned_actions > 0) {
          cat(sprintf("[Depth = %d] Pruned %d actions\n", depth, num_pruned_actions))
        }
      } else {
        private$action_list %<>% 
          # mutate(value = lower)
          mutate(value = map_dbl(action, ~ .x$expand(belief, discount, depth)))
      }
      
      private$action_list %<>% 
        arrange(desc(value))
      
      private$expanded <- TRUE
       
      return(self$value)
    },
    
    calculate_mean_max_reward = function(belief) {
      private$action_list$action %>% 
        map(~ .x$calculate_state_rewards(belief)) %>% 
        transpose() %>% 
        map_dbl(~ max(unlist(.x))) %>% 
        mean()
    },
    
    get_next_action_set = function(last_action) stop("Not implemented."),
    
    calculate_best_true_reward = function(belief) { map_dbl(private$action_list$action, ~ .x$calculate_true_reward(belief)) %>% max() },
    
    get_true_best_action = function(belief) { 
      private$action_list %>% 
        mutate(true_reward = map_dbl(action, ~ .x$calculate_true_reward(belief))) %>% 
        arrange(desc(true_reward)) %$% 
        first(action)
    }
  ),

  active = list(
    value = function() if (private$expanded) first(private$action_list$value),
    best_action = function() if (private$expanded) first(private$action_list$action)
  ),

  private = list(
    expanded = FALSE,
    action_list = NULL
  )
)

load_belief_node <- function(file, ...) pickleR::unpickle(file, ...)

BeliefNode <- R6Class(
  "BeliefNode",
  
  public = list(
    initialize = function(current_program_beliefs, prior_belief, action_set = NULL) {
      private$current_program_beliefs <- current_program_beliefs
      private$prior_belief_obj <- prior_belief
      private$action_set_obj <- action_set
      private$belief_period <- if (!is_null(prior_belief)) prior_belief$period + 1 else 1
    
      stopifnot(all(!map_lgl(private$current_program_beliefs, is_null)))
    },
    
    expand = function(discount, depth) {
      private$action_set_obj$expand(self, discount, depth)
    },
    
    calculate_expected_reward = function(treated_programs) {
      imap_dbl(private$current_program_beliefs,
               function(program_belief, program_index) program_belief$calculate_expected_reward(program_index %in% treated_programs)) %>% 
        sum()
    },
    
    calculate_state_rewards = function(treated_programs) {
      imap(private$current_program_beliefs, function(program_belief, program_index) program_belief$calculate_state_rewards(program_index %in% treated_programs)) %>% 
        reduce(add)
    },
    
    calculate_expected_reward_range = function(treated_programs, width = 0.8) {
      imap_dfr(private$current_program_beliefs,
               function(program_belief, program_index) program_belief$calculate_expected_reward_range(program_index %in% treated_programs, width)) %>% 
        colSums()
    },
    
    get_sampled_future_beliefs = function(action, ...) {
      sampled_program_beliefs <- private$current_program_beliefs %>%
        imap(function(program_belief, program_index, evaluated_programs, num_sim, ...) {
          if (program_index %in% evaluated_programs) {
            program_belief$get_simulated_beliefs(num_sim, ...)
          } else {
            map(seq(num_sim), ~ program_belief) # rep() doesn't work on env
          }
        }, evaluated_programs = action$evaluated_programs, num_sim = min(map_int(private$current_program_beliefs, ~ .x$num_simulated_samples)),...) %>%
        transpose()
     
      sampled_program_beliefs %>% 
        map(BeliefNode$new, prior_belief = self, action_set = private$action_set_obj$get_next_action_set(action))
    },
    
    save = function(file, ...) pickleR::pickle(self, file, ...)
  ),
  
  active = list(
    action_set = function() private$action_set_obj,
    value = function() self$action_set$value,
    program_beliefs = function() private$current_program_beliefs,
    best_action = function() self$action_set$best_action,
    period = function() private$belief_period,
    prior_belief = function() private$prior_belief_obj,
    
    reward_param = function() {
      map_dfr(private$current_program_beliefs, ~ .x$reward_param, .id = "program_id") %>% 
        mutate(period_id = private$belief_period)
    },
    
    model_param_draws = function() {
      map_dfr(private$current_program_beliefs, ~ .x$model_param_draws, .id = "program_id") %>% 
        mutate(period_id = private$belief_period)
    },
    
    all_periods_reward_param = function() {
      if (!is_null(private$prior_belief_obj)) {
        bind_rows(private$prior_belief_obj$all_periods_reward_param, self$reward_param)
      } else {
        self$reward_param
      }
    }
  ),
  
  private = list(
    action_set_obj = NULL,
    current_program_beliefs = NULL,
    prior_belief_obj = NULL,
    belief_period = 1
  )
)

ObservedBeliefNode <- R6Class(
  "ObservedBeliefNode",
  inherit = BeliefNode,

  public = list(
    execute_best_action = function(current_period, ...) {
      self$best_action$execute(self, current_period, ...)
    },
   
    calculate_true_reward = function(treated_programs) {
      imap_dbl(
        private$current_program_beliefs,
        function(observed_program_belief, program_index) observed_program_belief$program_period$get_reward(program_index %in% treated_programs)
      ) %>% 
        sum()
    },
    
    print_optimal_trajectory = function() {
      if (is_null(self$prior_belief)) {
        return(self$best_action$print_simple(self))
      } else {
        return(str_c(self$prior_belief$print_optimal_trajectory(), if (!is_null(self$best_action)) self$best_action$print_simple(self), sep = ", "))
      }
    }
  ),
  
  active = list(
    optimal_trajectory_data = function() {
      if (is_null(self$prior_belief)) {
        return(NULL)
      } else {
        node_optim_data <- lst(
          action = list(self$prior_belief$best_action),
          best_action = list(self$prior_belief$action_set$get_true_best_action(self)),
          reward = self$prior_belief$best_action$calculate_reward(self$prior_belief),
          value = self$prior_belief$action_set$value,
          true_reward = self$prior_belief$best_action$calculate_true_reward(self),
          best_true_reward = self$prior_belief$action_set$calculate_best_true_reward(self),
        )
        
        return(bind_rows(self$prior_belief$optimal_trajectory_data, node_optim_data))
      }
    }
  )
)

ProgramBelief <- R6Class(
  "ProgramBelief",
  
  public = list(
    initialize = function(dataset_draws,
                          is_simulated = TRUE,
                          prior_belief = NULL, 
                          hyperparam = if (!is_null(prior_belief)) prior_belief$hyperparam else stop("Hyperparameters not specified."), 
                          num_simulated_future_datasets = 1,
                          stan_model = if (!is_null(prior_belief)) prior_belief$stan_model else stop("Stan model not specified."), 
                          ...) {
      private$is_simulated_belief <- is_simulated
      private$hyperparam_list <-  hyperparam
      private$prior_belief_obj <- prior_belief
      private$stan_model_obj <- stan_model
      
      private$belief_dataset <- private$get_dataset_from_draws(dataset_draws) %>% 
        list_modify(study_size = length(.$y_control))
      
      stopifnot(with(private$belief_dataset, length(y_control) == length(y_treated)))
      
      belief_fit <- private$fit_to_data(hyperparam, ...) 
      
      all_draws <- posterior::as_draws_df(belief_fit)
    
      private$belief_draws <- all_draws %>% 
        tidybayes::spread_draws(new_mu_study, new_tau_study) %>% 
        ungroup() 
      
      private$simulated_future_draws <- all_draws %>% 
        sample_n(num_simulated_future_datasets) %>% 
        tidybayes::spread_draws(y_control_sim[period, obs_index], y_treated_sim[period, obs_index]) %>% 
        nest(sim_draws = c(period, obs_index, matches("y_(control|treated)_sim"))) 
      
      private$calculate_reward_param()
    },
    
    calculate_expected_reward = function(treated) with(slice(private$reward_param_mean, 1), new_mu_study + treated * new_tau_study), 
    
    calculate_state_rewards = function(treated) with(private$belief_draws, new_mu_study + treated * new_tau_study), 
    
    calculate_expected_reward_range = function(treated, width = 0.8) {
      filter(private$reward_param_mean, .width == width) %>% 
        with(c(lower = new_mu_study.lower + treated * new_tau_study.lower, upper = new_mu_study.upper + treated * new_tau_study.upper)) 
    },
    
    get_simulated_beliefs = function(num, hyperparam = NULL, ...) {
      hyperparam <- hyperparam %||% self$hyperparam 
      
      new_beliefs <- if (num > 1) {
        private$simulated_future_draws %>% 
          sample_n(num) %$%
          future_map(
            # BUG every so often I get an exception because of Stan fit with no samples. Not sure why.
            sim_draws, function(draws, ...) tryCatch(ProgramBelief$new(draws, ...), error = function(err) NULL), 
            is_simulated = TRUE, prior_belief = self, hyperparam = hyperparam, 
            # Here I'm only using multiple simulated datasets for the first set of simulated datasets, after that only one draw is used.
            num_simulated_future_datasets = 1,
            .options = furrr_options(
              packages = c("magrittr", "tidyverse"), 
              seed = TRUE,
              globals = c("ProgramBelief", "BeliefNode", "KBanditActionSet", "ActionSet", "Action")
            ),
            .progress = FALSE 
          ) %>% 
          compact()
      } else {
        private$simulated_future_draws %>% 
          sample_n(num) %$%
          map(
            sim_draws, ProgramBelief$new, 
            is_simulated = TRUE, prior_belief = self, hyperparam = hyperparam, 
            # Here I'm only using multiple simulated datasets for the first set of simulated datasets, after that only one draw is used.
            num_simulated_future_datasets = 1
          )
      }
      
      return(new_beliefs)
    }
 ),
  
  active = list(
    hyperparam = function() private$hyperparam_list,
    num_studies = function() if (is_null(private$prior_belief_obj)) 1 else private$prior_belief_obj$num_studies + 1,
    num_simulated_samples = function() nrow(private$simulated_future_draws),
    is_simulated = function() private$is_simulated_belief,
    prior_belief = function() private$prior_belief_obj,
    stan_model = function() private$stan_model_obj,
    reward_param = function() private$reward_param_mean,
    model_param_draws = function()draws$belief_draws,
    
    all_data = function() {
      if (!is_null(private$prior_belief_obj)) {
        list_merge(private$prior_belief_obj$all_data, !!!private$belief_dataset) %>% 
          list_modify(n_study = .$n_study + 1)
      } else {
        list_modify(private$belief_dataset, n_study = 1)
      }
    }
  ),
  
  private = list(
    hyperparam_list = NULL,
    belief_dataset = NULL,
    prior_belief_obj = NULL,
    simulated_future_draws = NULL,
    is_simulated_belief = TRUE,
    reward_param_mean = NULL,
    stan_model_obj = NULL,
    belief_draws = NULL,
    
    fit_to_data = function(hyperparam, generate_dataset_size = lst(n_control_sim = 50, n_treated_sim = 50), iter_warmup = 500, iter_sampling = iter_warmup, ...) {
      stan_data <- lst(
        fit = TRUE,
        sim = TRUE,
        sim_forward = TRUE,
        
        !!!generate_dataset_size,
        !!!self$all_data,
        !!!hyperparam
      ) %>% 
        list_modify(study_size = as.array(.$study_size))
      
      #   model$variational(data = stan_data)
      # tryCatch(
        # model$sample(data = stan_data, iter_warmup = iter_warmup, iter_sampling = iter_sampling, chains = 4, parallel_chains = 4, max_treedepth = 12, refresh = 0, show_messages = FALSE, ...),
        rstan::sampling(self$stan_model, data = stan_data, 
                        warmup = iter_warmup, iter = iter_warmup + iter_sampling, chains = 4, cores = 4, 
                        control = lst(max_treedepth = 12), 
                        refresh = 0, show_messages = TRUE, open_progress = FALSE, ...)
      #   error = function(err) browser()
      # )
    },
    
    get_dataset_from_draws = function(draws) {
      draws %>% 
        ungroup() %>% 
        select(y_control = y_control_sim, y_treated = y_treated_sim) %>% 
        as.list() 
    },
    
    calculate_reward_param = function() {
      private$reward_param_mean <- private$belief_draws %>% 
        tidybayes::mean_qi(.width = c(0.8)) 
      
      invisible(private$reward_param_mean)
    }
  )
)

ObservedProgramBelief <- R6Class(
  "ObservedProgramBelief",
  inherit = ProgramBelief,

  public = list(
    initialize = function(program_period, dataset_draws, ...) {
      super$initialize(dataset_draws, is_simulated = FALSE, ...)
      private$program_period_obj <- program_period
    }, 
    
    get_next_observed_belief = function(current_period, ...) {
      stopifnot(current_period > private$program_period_obj$period_index)
      private$program_period_obj$program$periods[[current_period]]$get_observed_belief(prior_belief = self, ...)
    }
  ),

  active = list(
    program_period = function() private$program_period_obj
  ),

  private = list(
    program_period_obj = NULL
  )
)