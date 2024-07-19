"This file contains all experimental quantities"

amplitude_reflectance_mirror1 = 0.99
amplitude_reflectance_mirror2 = 0.99
amplitude_transmittance_mirror1 = 0.01
amplitude_transmittance_mirror2 = 0.01
wavelength = 1064 * 10 ** (-9)  # lasers wavelength in meter
length_interferometer = 220865 * wavelength  # length is asummed to be a multiple of wavelength here 0.47m
speed_of_light = 299792458.0  # in meter per seconds
amplitude_input = 1000  # amplitude of input field
linewidth_mirror_1 = 1.99 * 10 ** 9  # Hz
linewidth_mirror_2 = 1.99 * 10 ** 9  # Hz
free_spectral_range = speed_of_light / (2*length_interferometer) # Hz
conversion_factor = 8.8  # V
frequency_laser = speed_of_light/wavelength

modulation_frequency= 7*10**6 #Hz


MaxTimesteps = 100
