library(R6)
library(magrittr)
library(tidyverse)
library(furrr)

rstan::rstan_options(auto_write = TRUE)

# stan_model <- cmdstanr::cmdstan_model("sim_model.stan")
stan_model <- rstan::stan_model("sim_model.stan")

# Classes -----------------------------------------------------------------

source("rl_environment.R")
source("rl_beliefs_and_actions.R")

utility <- function(outcome, alpha = 1.5) 1 - exp(- alpha * outcome)
expected_utility <- function(outcome_mean, outcome_sd, alpha = 1.5) 1 - exp(- alpha * outcome_mean + alpha^2 * outcome_sd^2 / 2)

KBanditActionSet <- R6Class(
  "KBanditActionSet", 
  inherit = ActionSet,

  public = list(
    initialize = function(k, choose = 1, last_action = NULL) {
      private$num_programs <- k
      private$choose_from_k <- choose
     
      combn(k, choose) %>% 
        plyr::alply(2, function(implement) Action$new(implement, action_set = self, last_action = last_action)) %>% 
        # map(seq(k), Action$new, action_set = self, last_action = last_action) %>% 
        super$initialize()
    },
    
    get_next_action_set = function(last_action) KBanditActionSet$new(self$k, self$choose, last_action) 
  ),

  active = list(
    k = function() private$num_programs,
    choose = function() private$choose_from_k,
    num_actions = function() nrow(private$action_list)
  ),
  
  private = list(
    num_programs = NULL,
    choose_from_k = 1
  )
)

# Solve -------------------------------------------------------------------

num_sim <- 6

testenv <- map(seq(num_sim), ~ { 
  create_environment(
    num_programs = 5, 
    num_periods = 10, 
    env_stan_model = stan_model,
    env_hyperparam = lst(
      mu_sd = 1,
      tau_mean = 0.1,
      tau_sd = 0.25,
      sigma_sd = 1,
      # eta_sd = c(1, 2, 1),
      eta_sd = c(0.1, 0.1, 0.1)
    ),
    expected_utility_fun = expected_utility
  )
})

testenv %>% map(~ .x$programs) %>% imap_dfr(~ map_dfr(.x, ~ .x$toplevel_params), .id = "sim_id")

infer_hyperparam = lst(
  mu_sd = 2,
  tau_mean = 0.05,
  tau_sd = 0.5,
  sigma_sd = 1.5,
  eta_sd = c(0.2, 0.2, 0.2),
)

run_rl_sim <- function(env, sim_id, depth, discount = 0.95) {
  plan(multisession, workers = 12)
  
  cat(sprintf("Simulation %d\n", sim_id))
  
  tryCatch({
      env$num_programs %>% 
        KBanditActionSet$new() %>% 
        env$solve_online_pomdp(
          hyperparam = infer_hyperparam, 
          num_simulated_future_datasets = 100, 
          stan_model = stan_model,
          discount = discount, 
          plan_depth = depth,
          iter_warmup = 1000
        )
    }, 
    error = function(err) { print(err); return(NULL) },
    finally = plan(sequential) 
  )
}

system.time(top_belief_node <- testenv %>% imap(run_rl_sim, depth = 2))

system.time(top_myopic_belief_node <- testenv %>% imap(run_rl_sim, depth = 0))

# Results -----------------------------------------------------------------

testenv$programs %>% map_dfr(~ .x$toplevel_params)

top_belief_node %>% 
  compact() %>% 
  map(~ .x$print_optimal_trajectory())

top_belief_node2 %>% 
  compact() %>% 
  map(~ .x$optimal_trajectory_data) %>% 
  map(mutate, 
      action_id = map_int(action, ~ .x$implemented_programs),
      best_action_id = map_int(best_action, ~ .x$implemented_programs))

top_myopic_belief_node %>% 
  compact() %>% 
  map(~ .x$print_optimal_trajectory())

top_myopic_belief_node2 %>% 
  compact() %>% 
  map(~ .x$optimal_trajectory_data) %>% 
  map(mutate, 
      action_id = map_int(action, ~ .x$implemented_programs),
      best_action_id = map_int(best_action, ~ .x$implemented_programs))

top_belief_node[[3]]$all_periods_reward_param %>% 
  ggplot() +
  geom_pointrange(
    aes(
      x = program_id, y = new_mu_study + new_tau_study, ymin = new_mu_study.lower + new_tau_study.lower, ymax = new_mu_study.upper + new_tau_study.upper
    )
  ) +
  facet_wrap(vars(period_id), nrow = 1) +
  theme_minimal()

top_myopic_belief_node2[[1]]$all_periods_reward_param %>% 
  ggplot() +
  geom_point(
    aes(
      x = program_id, 
      y = eu_control, 
      color = "Control"
    )
  ) +
  # geom_pointrange(
  #   aes(
  #     x = program_id, 
  #     y = expected_utility(new_mu_study, new_sigma_study), 
  #     ymin = expected_utility(new_mu_study.lower, new_sigma_study.lower), 
  #     ymax = expected_utility(new_mu_study.upper, new_sigma_study.upper),
  #     color = "Control"
  #   )
  # ) +
  # geom_pointrange(
  #   aes(
  #     x = program_id, y = new_mu_study + new_tau_study, ymin = new_mu_study.lower + new_tau_study.lower, ymax = new_mu_study.upper + new_tau_study.upper,
  #     color = "Treated"
  #   )
  # ) +
  facet_wrap(vars(period_id), nrow = 1) +
  theme_minimal()


trajectories <- lst(top_belief_node, top_myopic_belief_node) %>% 
  map_dfr(~ mutate(.x$optimal_trajectory_data, period = seq(n())), .id = "algo") %>% 
  mutate(
    program = map_int(action, ~ .x$evaluated_programs),
    across(c(true_reward, best_true_reward), ~ .x - .y, min(c(true_reward, best_true_reward))),
    true_reward_rate = true_reward / best_true_reward,
    run_id = 3
  ) %>% 
  select(!action) %>% 
  bind_rows(read_rds("temp/tbn.rds"))

trajectories %>% 
  ggplot(aes(as.integer(period))) +
  geom_line(aes(y = true_reward_rate, color = algo)) +
  facet_wrap(vars(run_id)) +
  scale_x_discrete()


# Save --------------------------------------------------------------------

trajectories %>% 
  write_rds("temp/tbn.rds") 
