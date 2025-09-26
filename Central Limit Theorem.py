import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Parameters
N = 50 # Size of each random vector
M = 10000 # Number of random vectors
# Function to generate means and plot distribution
def plot_means_and_verify_clt(distribution_name, samples, ax):
    means = np.mean(samples, axis=1)
    ax.hist(means, bins=50, density=True, alpha=0.7, label=f"Sampled {distribution_name}")
    # Fit a normal distribution to the means
    mu, std = norm.fit(means)
    x = np.linspace(min(means), max(means), 1000)
    ax.plot(x, norm.pdf(x, mu, std), label=f"Fitted Normal (\u03bc={mu:.2f}, \u03c3={std:.2f})")
    ax.set_title(f"{distribution_name} (N={N}, M={M})")
    ax.set_xlabel("Arithmetic Mean")
    ax.set_ylabel("Density")
    ax.legend()
# Subplot setup
fig, axs = plt.subplots( 1,3, figsize=(15,5 ))
axs = axs.ravel()
# (a) Binomial distribution
p = 0.5
binomial_samples = np.random.binomial(n=10, p=p, size=(M, N))
plot_means_and_verify_clt("Binomial", binomial_samples, axs[0])
# (b) Poisson distribution
lambda_param = 5
poisson_samples = np.random.poisson(lam=lambda_param, size=(M, N))
plot_means_and_verify_clt("Poisson", poisson_samples, axs[1])
# (c) Normal distribution
normal_samples = np.random.normal(loc=0, scale=1, size=(M, N))
plot_means_and_verify_clt("Normal", normal_samples, axs[2])
plt.tight_layout()
plt.show()
