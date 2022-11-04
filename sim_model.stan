
data {
  int<lower = 0, upper = 1> fit;
  int<lower = 0, upper = 1> sim;
  int<lower = 0, upper = 1> sim_forward;
 
  int n_control_sim;
  int n_treated_sim;
  int n_study;
  array[n_study] int study_size; // Assuming that the control and treatment arms are same size
  
  vector[fit ? sum(study_size) : 0] y_control;
  vector[fit ? sum(study_size) : 0] y_treated;
 
  // Hyperpriors 
  real<lower = 0> mu_sd;
  real tau_mean;
  real<lower = 0> tau_sd;
  real<lower = 0> sigma_sd;
  vector<lower = 0>[3] eta_sd;
}

parameters {
  real mu_toplevel;
  real tau_toplevel;
  real<lower = 0> sigma_toplevel; // Homoskedastic 
  vector<lower = 0>[3] eta_toplevel;
  
  vector[n_study] mu_study_raw;
  vector[n_study] tau_study_raw;
  vector[n_study] sigma_study_effect_raw;
}

transformed parameters {
  // Assuming these are drawn independently; identity correlation matrix
  vector[n_study] mu_study = mu_toplevel + mu_study_raw * eta_toplevel[1]; 
  vector[n_study] tau_study = tau_toplevel + tau_study_raw * eta_toplevel[2];
  vector<lower = 0>[n_study] sigma_study = sigma_toplevel * exp(sigma_study_effect_raw * eta_toplevel[3]);
}

model {
  mu_toplevel ~ normal(0, mu_sd);
  tau_toplevel ~ normal(tau_mean, tau_sd);
  sigma_toplevel ~ normal(0, sigma_sd);
  eta_toplevel ~ normal(0, eta_sd);
  
  mu_study_raw ~ std_normal();
  tau_study_raw ~ std_normal();
  sigma_study_effect_raw ~ std_normal();
  
  if (fit) {
    int study_pos = 1;
    
    for (study_index in 1:n_study) {
      int study_end = study_pos + study_size[study_index] - 1;
      
      y_control[study_pos:study_end] ~ lognormal(mu_study[study_index], sigma_study[study_index]);
      y_treated[study_pos:study_end] ~ lognormal(mu_study[study_index] + tau_study[study_index], sigma_study[study_index]);
      
      study_pos = study_end + 1;
    }
  }
}

generated quantities {
  array[sim ? (sim_forward ? 1 : n_study) : 0] vector[n_control_sim] y_control_sim;
  array[sim ? (sim_forward ? 1 : n_study) : 0] vector[n_treated_sim] y_treated_sim;

  if (sim) {
    if (sim_forward) {
      real new_mu_study = normal_rng(mu_toplevel, eta_toplevel[1]);
      real new_tau_study = normal_rng(tau_toplevel, eta_toplevel[2]);
      real new_sigma_study = sigma_toplevel * exp(normal_rng(0, eta_toplevel[3]));
      
      y_control_sim[1] = to_vector(lognormal_rng(rep_vector(new_mu_study, n_control_sim), rep_vector(new_sigma_study, n_control_sim)));
      y_treated_sim[1] = to_vector(lognormal_rng(rep_vector(new_mu_study + new_tau_study, n_treated_sim), rep_vector(new_sigma_study, n_treated_sim)));
    } else {
      for (study_index in 1:n_study) {
        y_control_sim[study_index] = to_vector(lognormal_rng(rep_vector(mu_study[study_index], n_control_sim), rep_vector(sigma_study[study_index], n_control_sim)));
        y_treated_sim[study_index] = to_vector(lognormal_rng(rep_vector(mu_study[study_index] + tau_study[study_index], n_treated_sim), rep_vector(sigma_study[study_index], n_treated_sim)));
      }
    }
  }
}
