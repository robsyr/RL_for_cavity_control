def power_output(delta_L):
    omega = 0
    # calculate the amplitude output , omega is the frequency added to carrier frequency here 0 because we assume system is in equilibrium
    phi = (
                  2 * np.pi / wavelength * (speed_of_light)) * (
                  length_interferometer + delta_L) / speed_of_light

    carrier_through_mirrors = -amplitude_transmittance_mirror1 * amplitude_transmittance_mirror2 * np.exp(
        1j * phi) / (1 - amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
        2j * phi)) * amplitude_input

    """ ATM we deal with the transmitted power, since the algos seem to work way better with output than with no output

    reflected_carrier = (amplitude_reflectance_mirror1-amplitude_reflectance_mirror2*np.exp(2j*phi))/(1-amplitude_reflectance_mirror1*amplitude_reflectance_mirror2*np.exp(2j*phi))*amplitude_input

    """
    power = np.float32(np.abs(carrier_through_mirrors) ** 2)

    return power