import gymnasium as gym
import numpy as np
import random
import scipy as sc
import matplotlib.pyplot as plt

from experimental_quantities import *


def error_signal(frequency, modulation_frequency, beta, input_power):
    r = amplitude_reflectance_mirror2
    P_carrier = sc.special.jv(0, beta) ** 2 * input_power
    P_sidebands = sc.special.jv(1, beta) ** 2 * input_power
    FSR = free_spectral_range
    term_1 = 2 * r ** 2 / FSR * np.sin(frequency / FSR) / (1 + r ** 4 - r ** 2 * np.cos(frequency / FSR))
    term_2 = r ** 2 * (2 - 2 * np.cos(frequency / FSR)) / (
                1 + r ** 4 - r ** 2 * np.cos(frequency / FSR)) ** 2 * r ** 2 / FSR * np.sin(frequency / FSR)
    Ableitung = term_1 - term_2
    return 2*np.sqrt(P_carrier * P_sidebands) * Ableitung * modulation_frequency


a = -0.7
b = 0.7

frequency = np.linspace(a * free_spectral_range, b * free_spectral_range, num=1000)
labels_list = np.linspace(a * free_spectral_range, b * free_spectral_range, num=6)
ticks = (labels_list / free_spectral_range).round(2)
error = error_signal(frequency, modulation_frequency, 0.5, 0.5)


coefficients = np.polyfit(frequency, error, 1)
m=coefficients[0]
b=coefficients[1]
y= m*frequency+b

def inverse_linear(Noise):
    function_value = Noise/13.6 *0.04  # 0.04 is the gesamthöhe des Fehlersignals
    x= (function_value-b)/m
    return x

print(inverse_linear(0.00029411764))

def frequency_to_length_difference(frequency):
    resonance_frequency= speed_of_light/wavelength
    length = speed_of_light/(resonance_frequency+frequency)
    difference = length- wavelength
    return difference

print(frequency_to_length_difference(inverse_linear(0.2)))

plt.plot(frequency, error)
plt.plot(frequency, y)
plt.xticks(labels_list, ticks)

plt.show()

# Use lambda to create a function that takes only one argument (the frequency)
error_at_freq = lambda x: error_signal(x, modulation_frequency, 0.51, 0.5)-0.00002941176

# Find the root of the error function using scipy.optimize.newton
root = sc.optimize.newton(error_at_freq, 0)

print("Root of error function:", root)