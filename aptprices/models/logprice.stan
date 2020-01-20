data {
  int N;
  int M;  // number of zip codes
  int M1;  // number of area1 codes
  int M2;  // number of area2 codes
  vector[N] log_price;
  int<lower=1> count[N];  // number of transactions
  real t[N];   // coded unit time
  real d[N];   // population density
  int zip_ind[N];  // zip index
  int area1_ind[N];  // area1 index
  int area2_ind[N];  // area2 index
}
transformed data {
  matrix[N, 6] X; // design matrix
  vector[6] w0;  // weights prior mean
  for (i in 1:N) { X[i, 1] = 1; X[i, 2] = t[i]; X[i, 3] = t[i]*t[i];
    X[i, 4] = d[i]; X[i, 5] = t[i]*d[i]; X[i, 6] = t[i]*t[i]*d[i]; }
  for (i in 1:6) w0[i] = 0;
}
parameters {
  cholesky_factor_corr[3] L_Omega;
  cholesky_factor_corr[6] L_Omega1;
  cholesky_factor_corr[6] L_Omega2;
  vector<lower=0>[3] tau;
  vector<lower=0>[6] tau1;
  vector<lower=0>[6] tau2;
  matrix[M, 3] w;
  matrix[M1, 6] w1;
  matrix[M2, 6] w2;
  row_vector[6] w_mean;
  real<lower=0> sigma;
  real<lower=0> y_sigma;
  real<lower=0> df;
}
model {
  vector[N] obs_mean;
  vector[N] obs_sigma;
  row_vector[6] x;
  matrix[3, 3] L_Sigma_w;
  matrix[6, 6] L_Sigma_w1;
  matrix[6, 6] L_Sigma_w2;
  L_Sigma_w = diag_pre_multiply(tau, L_Omega);
  L_Sigma_w1 = diag_pre_multiply(tau1, L_Omega1);
  L_Sigma_w2 = diag_pre_multiply(tau2, L_Omega2);
  tau ~ lognormal(-2., 1.);
  tau1 ~ lognormal(-2., 1.);
  tau2 ~ lognormal(-2., 1.);
  L_Omega ~ lkj_corr_cholesky(2);
  L_Omega1 ~ lkj_corr_cholesky(2);
  L_Omega2 ~ lkj_corr_cholesky(2);
  w_mean ~ normal(0, 5);
  for (i in 1:M) w[i] ~ multi_normal_cholesky(head(w0, 3), L_Sigma_w);
  for (i in 1:M1) w1[i] ~ multi_normal_cholesky(w0, L_Sigma_w1);
  for (i in 1:M2) w2[i] ~ multi_normal_cholesky(w0, L_Sigma_w2);
  for (i in 1:N) {
    x = X[i];
    obs_mean[i] = x * (w_mean + w1[area1_ind[i]] + w2[area2_ind[i]])' + head(x, 3) * w[zip_ind[i]]';
    obs_sigma[i] = sqrt(y_sigma^2 + sigma^2 / count[i]);
  }
  sigma ~ normal(0, 2);
  y_sigma ~ normal(0, 2);
  df ~ normal(0, 20);
  log_price ~ student_t(df + 1, obs_mean, obs_sigma);
}
