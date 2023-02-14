functions {
  // vector utility(vector outcome, real alpha) {
  //   return 1 - exp(- alpha * outcome);
  // }
  // 
  // vector expected_utility(real alpha, vector outcome_mean, vector outcome_sd) {
  //   return 1 - exp(- alpha * outcome_mean + alpha^2 * outcome_sd^2 / 2);
  // }
}

data {
  int<lower = 0, upper = 1> fit;
  int<lower = 0, upper = 1> sim;
  int<lower = 0, upper = 1> sim_forward;
  int<lower = 0, upper = 1> sigma_eta_inv_gamma_priors;
 
  int n_control_sim;
  int n_treated_sim;
  int n_study;
  array[n_study] int study_size; // Assuming that the control and treatment arms are same size
  
  vector[fit ? sum(study_size) : 0] y_control;
  vector[fit ? sum(study_size) : 0] y_treated;
  
  // real alpha_utility;
 
  // Hyperpriors 
  real<lower = 0> mu_sd;
  real tau_mean;
  real<lower = 0> tau_sd;
  real<lower = 0> sigma_sd;
  vector<lower = 0>[3] eta_sd;
  real<lower = 0> sigma_alpha;
  real<lower = 0> sigma_beta;
  real<lower = 0> eta_alpha;
  real<lower = 0> eta_beta;
}

parameters {
  real mu_toplevel;
  real tau_toplevel;
  real<lower = 0> sigma_toplevel; // Homoskedastic 
  vector<lower = 0>[2] eta_toplevel;
  
  vector[n_study] mu_study_raw;
  vector[n_study] tau_study_raw;
  // vector[n_study] sigma_study_effect_raw;
}

transformed parameters {
  // Assuming these are drawn independently; identity correlation matrix
  vector[n_study] mu_study = mu_toplevel + mu_study_raw * eta_toplevel[1]; 
  vector[n_study] tau_study = tau_toplevel + tau_study_raw * eta_toplevel[2];
  vector<lower = 0>[n_study] sigma_study = rep_vector(sigma_toplevel, n_study); // * exp(sigma_study_effect_raw * eta_toplevel[3]);
  
  // vector[n_study] expected_utility_control = expected_utility(alpha_utility, mu_study, sigma_study); 
  // vector[n_study] expected_utility_treated = expected_utility(alpha_utility, mu_study + tau_study, sigma_study); 
}

model {
  mu_toplevel ~ normal(0, mu_sd);
  tau_toplevel ~ normal(tau_mean, tau_sd);
  
  if (sigma_eta_inv_gamma_priors) {
    sigma_toplevel ~ inv_gamma(sigma_alpha, sigma_beta);
    eta_toplevel ~ inv_gamma(eta_alpha, eta_beta);
  } else {
    sigma_toplevel ~ normal(0, sigma_sd);
    eta_toplevel ~ normal(0, eta_sd[:2]);
  }
  
  mu_study_raw ~ std_normal();
  tau_study_raw ~ std_normal();
  // sigma_study_effect_raw ~ std_normal();
  
  if (fit) {
    int study_pos = 1;
    
    for (study_index in 1:n_study) {
      int study_end = study_pos + study_size[study_index] - 1;
      
      // y_control[study_pos:study_end] ~ lognormal(mu_study[study_index], sigma_study[study_index]);
      // y_treated[study_pos:study_end] ~ lognormal(mu_study[study_index] + tau_study[study_index], sigma_study[study_index]);
      
      y_control[study_pos:study_end] ~ normal(mu_study[study_index], sigma_study[study_index]);
      y_treated[study_pos:study_end] ~ normal(mu_study[study_index] + tau_study[study_index], sigma_study[study_index]);
      
      study_pos = study_end + 1;
    }
  }
}

generated quantities {
  array[sim ? (sim_forward ? 1 : n_study) : 0] vector[n_control_sim] y_control_sim;
  array[sim ? (sim_forward ? 1 : n_study) : 0] vector[n_treated_sim] y_treated_sim;
  
  real new_mu_study = 0; 
  real new_tau_study = 0;
  real<lower = 0> new_sigma_study = 0; 
  
  // real new_expected_utility_control = 0;
  // real new_expected_utility_treated = 0;
  
  if (sim) {
    if (sim_forward) {
      new_mu_study = normal_rng(mu_toplevel, eta_toplevel[1]);
      new_tau_study = normal_rng(tau_toplevel, eta_toplevel[2]);
      new_sigma_study = sigma_toplevel; // * exp(normal_rng(0, eta_toplevel[3]));
      
      // new_expected_utility_control = expected_utility(alpha_utility, [ new_mu_study ]', [ new_sigma_study ]')[1]; 
      // new_expected_utility_treated = expected_utility(alpha_utility, [ new_mu_study + new_tau_study ]', [ new_sigma_study ]')[1]; 
      
      // y_control_sim[1] = to_vector(lognormal_rng(rep_vector(new_mu_study, n_control_sim), rep_vector(new_sigma_study, n_control_sim)));
      // y_treated_sim[1] = to_vector(lognormal_rng(rep_vector(new_mu_study + new_tau_study, n_treated_sim), rep_vector(new_sigma_study, n_treated_sim)));
      y_control_sim[1] = to_vector(normal_rng(rep_vector(new_mu_study, n_control_sim), rep_vector(new_sigma_study, n_control_sim)));
      y_treated_sim[1] = to_vector(normal_rng(rep_vector(new_mu_study + new_tau_study, n_treated_sim), rep_vector(new_sigma_study, n_treated_sim)));
    } else {
      for (study_index in 1:n_study) {
        // y_control_sim[study_index] = to_vector(lognormal_rng(rep_vector(mu_study[study_index], n_control_sim), rep_vector(sigma_study[study_index], n_control_sim)));
        // y_treated_sim[study_index] = to_vector(lognormal_rng(rep_vector(mu_study[study_index] + tau_study[study_index], n_treated_sim), rep_vector(sigma_study[study_index], n_treated_sim)));
        y_control_sim[study_index] = to_vector(normal_rng(rep_vector(mu_study[study_index], n_control_sim), rep_vector(sigma_study[study_index], n_control_sim)));
        y_treated_sim[study_index] = to_vector(normal_rng(rep_vector(mu_study[study_index] + tau_study[study_index], n_treated_sim), rep_vector(sigma_study[study_index], n_treated_sim)));
      }
    }
  }
}
