import random
from experimental_quantities import wavelength

def gaussian_noise(noise_factor_of_wavelength):
    """gaussian_noise, takes standard deviation as input and returns a random number distributed along the model"""
    return random.uniform(-noise_factor_of_wavelength * wavelength,
                          noise_factor_of_wavelength * wavelength), noise_factor_of_wavelength

print([gaussian_noise(0.0002) for i in range(10)])