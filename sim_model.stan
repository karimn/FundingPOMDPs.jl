
data {
  int<lower = 0, upper = 1> fit;
  int<lower = 0, upper = 1> sim;
 
  int n_control;
  int n_treated;
  int n_study;
  
  vector[fit ? n_control : 0] y_control;
  vector[fit ? n_treated : 0] y_treated;
 
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
  real<lower = 0> sigma_toplevel;
  vector<lower = 0>[3] eta_toplevel;
  
  vector[n_study] mu_study_raw;
  vector[n_study] tau_study_raw;
  vector<lower = 0>[n_study] sigma_study_raw;
}

transformed parameters {
  vector[n_study] mu_study = mu_toplevel + mu_study_raw * eta_toplevel[1]; 
  vector[n_study] tau_study = tau_toplevel + tau_study_raw * eta_toplevel[2];
  vector<lower = 0>[n_study] sigma_study = sigma_toplevel + sigma_study_raw * eta_toplevel[3];
}

model {
  mu_toplevel ~ normal(0, mu_sd);
  tau_toplevel ~ normal(tau_mean, tau_sd);
  sigma_toplevel ~ normal(0, sigma_sd);
  eta_toplevel ~ normal(0, eta_sd);
  
  mu_study_raw ~ std_normal();
  tau_study_raw ~ std_normal();
  sigma_study_raw ~ std_normal();
  
  if (fit) {
    y_control ~ normal(mu_study[1], sigma_study[1]);
    y_treated ~ normal(mu_study[1] + tau_study[1], sigma_study[1]);
  }
}

generated quantities {
  vector[sim ? n_control : 0] y_control_sim;
  vector[sim ? n_treated : 0] y_treated_sim;

  if (sim) {
    y_control_sim = to_vector(normal_rng(rep_vector(mu_study[1], n_control), rep_vector(sigma_study[1], n_control)));
    y_treated_sim = to_vector(normal_rng(rep_vector(mu_study[1] + tau_study[1], n_control), rep_vector(sigma_study[1], n_control)));
  }
}
