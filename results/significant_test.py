import numpy as np
from scipy.stats import ttest_rel

alpha = 0.05

# F1 scores before and after fine-tuning
sample_1_f1_scores_before = np.array([0.9282, 0.8155, 0.9004])
sample_1_f1_scores_after = np.array([0.9268, 0.8037, 0.8954])

# Perform paired t-test
t_statistic, p_value = ttest_rel(sample_1_f1_scores_before, sample_1_f1_scores_after)

print("Paired t-test Sample 1 Results:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Interpretation of the results
if p_value < alpha:
    print("The difference in F1 scores before and after fine-tuning is statistically significant (reject null hypothesis).")
else:
    print("The difference in F1 scores before and after fine-tuning is not statistically significant (fail to reject null hypothesis).")

print("")

# F1 scores before and after fine-tuning
sample_2_f1_scores_before = np.array([0.6627, 0.5693, 0.6358])
sample_2_f1_scores_after = np.array([0.7601, 0.6114, 0.7172])

# Perform paired t-test
t_statistic, p_value = ttest_rel(sample_2_f1_scores_before, sample_2_f1_scores_after)

print("Paired t-test Sample 2 Results:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Interpretation of the results
if p_value < alpha:
    print("The difference in F1 scores before and after fine-tuning is statistically significant (reject null hypothesis).")
else:
    print("The difference in F1 scores before and after fine-tuning is not statistically significant (fail to reject null hypothesis).")

print("")

# F1 scores before and after fine-tuning
sample_3_f1_scores_before = np.array([0.0, 0.6061, 0.6061])
sample_3_f1_scores_after = np.array([0.0, 0.5443, 0.5443])

# Perform paired t-test
t_statistic, p_value = ttest_rel(sample_3_f1_scores_before, sample_3_f1_scores_after)

print("Paired t-test Sample 3 Results:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Interpretation of the results
if p_value < alpha:
    print("The difference in F1 scores before and after fine-tuning is statistically significant (reject null hypothesis).")
else:
    print("The difference in F1 scores before and after fine-tuning is not statistically significant (fail to reject null hypothesis).")