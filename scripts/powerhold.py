from fabry_perot_pm import *
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from experimental_quantities import *

# Set the parameters
time = MaxTimesteps*10
noise_level = 4e-12
sample_size = 5

# Initialize data storage with steps
steps = np.arange(1, time + 1)
df_noise = pd.DataFrame({"step": steps})

# Get the initial power reference
power_eq = power_output_transmitted_fabry_perot(0)

# Vectorized generation of noise and power calculation
for j in range(sample_size):
    print(f"Running sample {j}")
    
    # Generate cumulative noise using a Gaussian distribution
    noise = np.cumsum(np.random.normal(0.0, noise_level, time))
    
    # Compute power with added noise
    power_noise = power_output_transmitted_fabry_perot(noise) / power_eq
    print(type(power_noise), power_noise[:10])
    # Add the generated power noise to the dataframe
    df_noise[f'power{j}'] = power_noise

# Melt the dataframe for plotting
melted_df = df_noise.melt(id_vars='step', var_name='Powers', value_name='Value')

# Plot with fewer points to speed up rendering
plt.figure(figsize=(16, 9))
sns.lineplot(data=melted_df.iloc[::100], x="step", y="Value", errorbar="sd", label='No agent')  # Plot every 100th point
plt.show()
