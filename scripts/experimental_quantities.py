"""This file contains all experimental quantities"""

amplitude_reflectance_mirror1 = 0.99                                # reflectivity of mirror 1
amplitude_reflectance_mirror2 = 0.99                                # reflectivity of mirror 2
amplitude_transmittance_mirror1 = 0.01                              # transmitivity of mirror 1
amplitude_transmittance_mirror2 = 0.01                              # transmitivity of mirror 2
wavelength = 1064 * 10 ** (-9)                                      # lasers wavelength in meter
length_interferometer = 220865 * wavelength                         # length is asummed to be a multiple of wavelength here 0.47m
speed_of_light = 299792458.0                                        # speed of light in meter per seconds
amplitude_input = 1000                                              # amplitude of input field
linewidth_mirror_1 = 1.99 * 10 ** 9                                 # Linewidth of mirror 1 in Hz
linewidth_mirror_2 = 1.99 * 10 ** 9                                 # Linewidth of mirror 2 in Hz
free_spectral_range = speed_of_light / (2*length_interferometer)    # Free spectral range in Hz
conversion_factor = 8.8                                             # Conversion factor used for expermintal noise analysis in V
frequency_laser = speed_of_light/wavelength                         # frequency of laser in Hz

modulation_frequency= 7*10**6                                       # Moudulation Frequency of the laser in Hz


MaxTimesteps = 100                                                  # Maximal timesteps for Learning
