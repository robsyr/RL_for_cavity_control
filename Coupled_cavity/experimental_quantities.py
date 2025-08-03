"""This file contains all experimental quantities"""

amplitude_reflectance_mirror1 = 0.99  # reflectivity of mirror 1
amplitude_reflectance_mirror2 = 0.99  # reflectivity of mirror 2
amplitude_transmittance_mirror1 = 0.01  # transmitivity of mirror 1
amplitude_transmittance_mirror2 = 0.01  # transmitivity of mirror 2
wavelength = 1064 * 10 ** (-9)  # lasers wavelength in meter
length_interferometer = (
    220865 * wavelength
)  # length is asummed to be a multiple of wavelength here 0.47m
speed_of_light = 299792458.0  # speed of light in meter per seconds

MaxTimesteps = 200  # Maximal timesteps for Learning
