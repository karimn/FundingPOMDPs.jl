library(R6)
library(magrittr)
library(tidyverse)
library(furrr)

rstan::rstan_options(auto_write = TRUE)

# Main --------------------------------------------------------------------

source("rl_environment.R")
source("rl_beliefs_and_actions.R")
source("kbandit.R")

plan(multisession, workers = 12)

# stan_model <- cmdstanr::cmdstan_model("sim_model.stan")
stan_model <- rstan::stan_model("sim_model.stan")

testenv <- create_environment(
  num_programs = 5, 
  num_periods = 6, 
  env_stan_model = stan_model,
  env_hyperparam = lst(
    mu_sd = 2,
    tau_mean = 0.1,
    tau_sd = 1,
    sigma_sd = 2,
    # eta_sd = c(1, 2, 1),
    eta_sd = c(0.1, 0.2, 0.1),
  ) 
)

system.time(
  top_belief_node <- testenv$num_programs %>% 
    KBanditActionSet$new() %>% 
    testenv$solve_online_pomdp(
      hyperparam = lst(
        mu_sd = 2.5,
        tau_mean = 0.2,
        tau_sd = 1.5,
        sigma_sd = 2.5,
        eta_sd = c(0.2, 0.4, 0.2),
      ), 
      num_simulated_future_datasets = 10, 
      stan_model = stan_model,
      discount = 0.9, 
      plan_depth = 4,
      iter_warmup = 1000
    )
)

top_belief_node$print_optimal_trajectory()
top_belief_node$optimal_trajectory_data

testenv$programs %>% map_dfr(~ .x$toplevel_params)

top_belief_node$all_periods_reward_param %>% 
  ggplot() +
  geom_pointrange(
    aes(
      x = program_id, y = new_mu_study + new_tau_study, ymin = new_mu_study.lower + new_tau_study.lower, ymax = new_mu_study.upper + new_tau_study.upper
    )
  ) +
  facet_wrap(vars(period_id), nrow = 1) +
  theme_minimal()

top_belief_node$all_periods_reward_param %>% 
  ggplot() +
  geom_pointrange(
    aes(
      x = program_id, y = new_mu_study, ymin = new_mu_study.lower, ymax = new_mu_study.upper,
      color = "Control"
    )
  ) +
  geom_pointrange(
    aes(
      x = program_id, y = new_mu_study + new_tau_study, ymin = new_mu_study.lower + new_tau_study.lower, ymax = new_mu_study.upper + new_tau_study.upper,
      color = "Treated"
    )
  ) +
  facet_wrap(vars(period_id), nrow = 1) +
  theme_minimal()

top_myopic_belief_node <- testenv$num_programs %>% 
  KBanditActionSet$new() %>% 
  testenv$solve_online_pomdp(
    hyperparam = lst(
      mu_sd = 2.5,
      tau_mean = 0.2,
      tau_sd = 1.5,
      sigma_sd = 2.5,
      eta_sd = c(0.2, 0.4, 0.2),
    ), 
    num_simulated_future_datasets = 10, 
    stan_model = stan_model,
    discount = 0.9, 
    plan_depth = 0,
    iter_warmup = 1000
  )

lst(top_belief_node, top_myopic_belief_node) %>% 
  map_dfr(~ mutate(.x$optimal_trajectory_data, period = seq(n())), .id = "algo") %>% 
  mutate(
    across(c(true_reward, best_true_reward), ~ .x - .y, min(c(true_reward, best_true_reward))),
    true_reward_rate = true_reward / best_true_reward
  ) %>% 
  select(!action) %>% 
  write_rds("temp/tbn.rds") %>% 
  ggplot(aes(as.integer(period))) +
  geom_line(aes(y = true_reward_rate, color = algo)) +
  scale_x_discrete()

plan(sequential)

