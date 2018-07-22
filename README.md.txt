# Spam-Classification

Objective is to identify spam emails. The dataset consists of 1099 mails. After preprocessing, only the subject and content of the mails are retained. The dataset is then split 2 sets - train set and test set. The train set is again randomly sampled to prepare 5 further training sets just to repeat our experiments.

For classifying spam and legit mails, we use the Naive bayes classifier algorithm. The bayesian spam filter is designed and implemented for the following 4 cases.
a. Maximum likelihood estimation using multinomial distribution.
b. Maximum likelihood estimation using bernoulli distribution.
c. Bayesian parameter estimation where prior are assumed to be dirichlet distributed.
d. Bayesian parameter estimation where prior are assumed to be beta distributed.

The final performance is measured using AUC.
