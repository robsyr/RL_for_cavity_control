import matplotlib.pyplot as plt
import pandas as pd

from import_noise import *
from calculate_output_powers import power_output_transmitted_fabry_perot


max_episode_steps = 50
data = {"step": [i+1 for i in range(max_episode_steps)]}
df=pd.DataFrame(data)
for j in range(10):
    power = []
    length_difference = 0

    for i in range(max_episode_steps):
        noise= gaussian_noise(noise_factor_of_wavelength=0.0002)[0]
        print(noise)
        powe_equilibrium =power_output_transmitted_fabry_perot(0)
        power_noise=power_output_transmitted_fabry_perot(noise + length_difference)
        length_difference+=noise
        #print(length_difference)
        power.append(np.abs(power_noise))
    df[f'power{j}']=power

melted_df = df.melt(id_vars='step', var_name='Powers', value_name='Value')


plt.figure(figsize=(16,9))
sns.lineplot(data=melted_df, x="step", y="Value", errorbar="sd")
plt.show()


print(power)