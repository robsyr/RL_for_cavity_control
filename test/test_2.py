import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# Define the constants
sigma = 1  # standard deviation for the Gaussian distribution
a, b = -10, 10  # lower and upper bounds for the Uniform distribution
x = 2  # the value at which to evaluate the PDFs

# Define a range of mean values for the Gaussian distribution
mean_values = np.linspace(-5, 5, 100)

# Calculate the negative log ratio for each mean value
neg_log_ratios = []
for mu in mean_values:
    gaussian_pdf = norm.pdf(x, loc=mu, scale=sigma)
    uniform_pdf = uniform.pdf(x, loc=a, scale=(b-a))
    ratio = gaussian_pdf / uniform_pdf
    neg_log_ratio = -np.log(ratio)
    neg_log_ratios.append(neg_log_ratio)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(mean_values, neg_log_ratios-min(neg_log_ratios), label='Negative Log Ratio')
plt.xlabel('Mean (μ)')
plt.ylabel('Negative Log Ratio')
plt.title('Negative Log Ratio of Gaussian vs Uniform Distribution')
plt.legend()
plt.grid(True)
plt.show()
