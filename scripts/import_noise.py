import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import copy


from experimental_quantities import *

# Assuming your CSV file is named 'your_file.csv' and you want to start reading from row 5
# You can adjust the path and skiprows parameter as needed

# Read CSV file into a DataFrame starting from row 5
df = pd.read_csv('C:\\Users\\robin\Documents\\Uni\MA\Rauschmessung\data20_03_13_11_36.csv', skiprows=68,
                 sep='\s+')  # Skip the first 4 rows (0-based indexing)
dataframe = copy.copy(df)
# dataframe['channel_4_V'] = dataframe['channel_4_V'].mul(2/(8.8) * 1064*10**(-9))
# df= df['channel_4_V'].mul(1/(8.8) * 1064*10**(-10))
# Display the DataFrame
# sns.histplot(data=df, x='channel_4_V', bins=20)
# plt.show()

'''We obtained a value which has to be a factor 100 smaller than described by mail'''



def signal_to_noise_conversion(df):
    df['frequency_change'] = df['channel_4_V'].diff().fillna(0).mul(
        free_spectral_range *50 / 0.173)  # take the difference of the absolut values of piezo
    df['new_resonance_frequency'] = frequency_laser+df['frequency_change']
    df['new_resonance_wavelength']=speed_of_light/(df['new_resonance_frequency'])
    df['length_difference'] = wavelength-df['new_resonance_wavelength']

    return df['length_difference']

"""
test=(signal_to_noise_conversion(df))
print(max(test), min(test))
print(max(df['channel_4_V']), min(df['channel_4_V']))
print(max(df['channel_1_V']), min(df['channel_1_V']), np.median(df['channel_1_V']))

df['normalized_power']=df['channel_1_V'].mul(1.0/np.mean(df['channel_1_V']))
print(df)
print(max(df['normalized_power']), min(df['normalized_power']))
"""

def noise_from_experiment(df_series):
    # create random number
    random_number = random.randint(0, 374999)
    difference_in_length = df_series[random_number]
    return difference_in_length


print(signal_to_noise_conversion(df).to_string())
#noise= [noise_from_experiment(signal_to_noise_conversion(df)) for i in range(100)]
#print(noise)

"""
farbrynoise=[noise_from_experiment(df) for i in range(10)]

print(f'FP:{farbrynoise}')
"""


def gaussian_noise(noise_factor_of_wavelength):
    """gaussian_noise, takes standard deviation as input and returns a random number distributed along the model"""
    return random.uniform(-noise_factor_of_wavelength * wavelength,
                          noise_factor_of_wavelength * wavelength), noise_factor_of_wavelength

print([gaussian_noise(0.0002) for i in range(10)])



