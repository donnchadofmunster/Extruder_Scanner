import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Define Classes ---------- #

class Laser: # The 905nm line laser
    def __init__(
            self, 
            output_power_watts,
            fan_angle_degrees,
            distance_to_target_meters, 
            ):

        self.output_power_watts = output_power_watts
        self.fan_angle_degrees = fan_angle_degrees    # Used alongside the distance to target to calculate line length
        self.distance_to_target_meters = distance_to_target_meters
        self.wavelength_meters = 905e-9  # 905 nm laser
        
        # Indirect Laser Parameters
        self.line_length_meters = 2 * self.distance_to_target_meters * math.tan(math.radians(self.fan_angle_degrees) / 2)
        self.standard_deviation_m = self.line_length_meters / 3.92  # 68-95-99.7 Rule, FWHM = 2.355σ, assume FWHM is around 1.67x line length
        self.irradiance = self.output_power_watts / (self.line_length_meters * self.standard_deviation_m * math.sqrt(2*math.pi))

class Camera: # The camera used to capture the laser line
    def __init__(
            self, 
            bit_depth,
            frames_per_second,
            pixel_size_m,
            quantum_efficiency,
            full_well_capacity_electrons # The maximum number of electrons a pixel can hold before oversaturating
            ):
        self.bit_depth = bit_depth
        self.frames_per_second = frames_per_second
        self.exposure_time_seconds = 1 / self.frames_per_second
        self.pixel_area_m2 = pixel_size_m**2
        self.quantum_efficiency = quantum_efficiency
        self.full_well_capacity_electrons = full_well_capacity_electrons
        self.maximum_digital_value = (2 ** self.bit_depth) - 1

class NDFilter: # The ND filter used to reduce laser intensity on the camera
    def __init__(
            self,
            optical_density = 0,
            ):
        self.optical_density = optical_density
        self.transmission_fraction = 10 ** (-self.optical_density)

class LED: # The LED array used for inside illumination
    def __init__(
            self,
            output_power_watts,
            ):
        self.output_power_watts = output_power_watts

        # Indirect LED Parameters
        self.irradiance = output_power_watts # PLACEHOLDER FOR NOW

class PEEKBeam: # The PEEK beam being extruded and scanned
    def __init__(
            self,
            wall_thickness_m,
            diameter_m,
            ):
        self.wall_thickness_m = wall_thickness_m
        self.diameter_m = diameter_m
        self.cross_sectional_area_m2 = math.pi * (self.diameter_m / 2) ** 2

# ---------- Define Functions ---------- #

def electrons_to_digital_pixel_value(
        electrons_at_pixel,
        camera: Camera,
        ) -> float:
    
    """
    Convert the number of electrons at a pixel to a digital pixel value based on the camera's full well capacity.
    The pixel value is found by normalising the number of electrons to the full well capacity and scaling it to the maximum
    digital value based on the camera's bit depth.

    Parameters:
    - electrons_at_pixel: Number of electrons at the pixel
    - camera: Camera object

    Returns:
    - Digital pixel value (eg. 0-255 for 8-bit camera)
    """

    pixel_value = (electrons_at_pixel / camera.full_well_capacity_electrons) * camera.maximum_digital_value
    pixel_value = min(max(pixel_value, 0), camera.maximum_digital_value) # Clamping, stops it from going below 0 or above maximum

    return pixel_value

# Crystallinity

def calculate_electrons_peak( 
        laser: Laser,
        camera: Camera,
        nd_filter: NDFilter,
        reflectivity = 0.5 # Maximum expected reflectivity is 50%
        ) -> float:
    
    """
    Calculate the number of electrons at the pixel at the centre of the gaussian curse, the one that will be used
    for calculating the crystallinity and thus tensile strength of the PEEK sample. The reason the one at the
    centre is used is because it will have the highest irradiance and thus the best signal to noise ratio.
    
    Parameters:
    - laser: Laser object
    - camera: Camera object
    - reflectivity: Reflectivity of from the PEEK surface, estimated to be between 20% to 50%
    
    Returns:
    - The number of electrons generated per pixel during the exposure time
    """

    h = 6.626e-34 # Planck's constant
    c = 3e8 # Speed of light

    photon_energy_joules = h * c / laser.wavelength_meters # E = hc/λ, 
    irradiance_reflected = laser.irradiance * reflectivity * nd_filter.transmission_fraction # Amount of light reflected off of the PEEK surface
    photons_flux = irradiance_reflected / photon_energy_joules # AKA photons per second per square meter
    photons_at_pixel = photons_flux * camera.pixel_area_m2 * camera.exposure_time_seconds # Photons hitting the pixel during the exposure time
    electrons_at_pixel = photons_at_pixel * camera.quantum_efficiency # Electrons generated by the pixel

    return electrons_at_pixel

def calculate_peak_pixel_value(  
        laser: Laser,
        camera: Camera,
        ndfilter: NDFilter,
        reflectivity = 0.5 # Maximum expected reflectivity is 50%
        ) -> float:
    
    """
    Combines the two functions above to calculate the pixel value directly from the laser line
    
    Parameters:
    - laser: Laser object
    - camera: Camera object
    - reflectivity: Reflectivity of from the PEEK surface, estimated to be between 20% to 50%

    Returns:
    - Digital pixel value (eg. 0-255 for 8-bit camera)
    """

    electrons_at_pixel = calculate_electrons_peak(laser, camera, ndfilter, reflectivity)
    pixel_value = electrons_to_digital_pixel_value(electrons_at_pixel, camera)

    return pixel_value

def pixel_value_to_reflectivity( 
        pixel_value,
        camera: Camera,
        laser: Laser,
        ndfilter: NDFilter,
        ) -> float:
    
    """
    Inverse of the above functions, converts a pixel value back to reflectivity

    Parameters:
    - pixel_value: Digital pixel value (eg. 0-255 for 8-bit camera)
    - camera: Camera object
    - laser: Laser object

    Returns:
    - Reflectivity value (0 to 1)
    """

    electrons_at_pixel = (pixel_value / camera.maximum_digital_value) * camera.full_well_capacity_electrons

    h = 6.626e-34 # Planck's constant
    c = 3e8 # Speed of light

    photon_energy_joules = h * c / laser.wavelength_meters # E = hc/λ, 
    photons_at_pixel = electrons_at_pixel / camera.quantum_efficiency # Photons hitting the pixel during the exposure time
    photons_flux = photons_at_pixel / (camera.pixel_area_m2 * camera.exposure_time_seconds) # Photons per second per square meter
    irradiance_reflected = photons_flux * photon_energy_joules
    reflectivity = irradiance_reflected / (laser.irradiance * ndfilter.transmission_fraction)

    return reflectivity

def reflectivity_to_crystallinity( 
        reflectivity
        ) -> float:
    
    """
    Inverse of the above functions, converts a pixel value back to reflectivity

    Parameters:
    - Reflectivity value (0 to 1)

    Returns:
    - The crystallinity percentage of the PEEK sample
    """

    reflectivity *= 100 # Convert to percentage
    reflectivity = max(reflectivity, 1e-6)
    if reflectivity <= 38.1: # Use linear interpolation to convert reflectivity to crystallinity
        crystallinity = 6.5 + 269.93*math.log10(reflectivity/31.5)
    else:
        crystallinity = 28.8 + 213.44*math.log10(reflectivity/38.8)

    return crystallinity

def crystallinity_to_mechanical( 
        crystallinity,
        ) -> float:
    
    """
    Convert crystallinity to tensile strength using linear interpolation from provided data arrays. The data
    is based on experimental results for PEEK at 905nm laser wavelength and was found from literature. The
    data is taken through interpolation to allow for a wider range of crystallinity values, however the accuracy
    is not guarenteed.

    Parameters:
    - crystallinity: The crystallinity percentage of the PEEK sample
    - crystallinity_array: Array of known crystallinity percentages
    - strength_array: Array of known tensile strengths corresponding to the crystallinity percentages

    Returns:
    - The tensile strength in MPa
    """

    crystallinity_array = [20, 21.7, 29.8, 30.5, 31] 
    strength_array = [58, 66, 74, 84, 83]
    elastic_modulus_array = [64.5, 65, 67, 83, 88]
    tensile_strength_mpa = np.interp(crystallinity, crystallinity_array, strength_array)
    elastic_modulus_gpa = np.interp(crystallinity, crystallinity_array, elastic_modulus_array)
    return tensile_strength_mpa, elastic_modulus_gpa

def pixel_value_to_mechanical( 
        pixel_value,
        laser: Laser,
        camera: Camera,
        ndfilter: NDFilter,
        ) -> float:
    
    """
    Combines the above functions to convert a pixel value directly to tensile strength and elastic modulus

    Parameters:
    - pixel_value: Digital pixel value (eg. 0-255 for 8-bit camera)
    - camera: Camera object
    - laser: Laser object

    Returns:
    - The tensile strength in MPa and elastic modulus in GPa
    """

    reflectivity = pixel_value_to_reflectivity(pixel_value, camera, laser, ndfilter)
    crystallinity = reflectivity_to_crystallinity(reflectivity)
    tensile_strength_mpa, elastic_modulus_gpa = crystallinity_to_mechanical(crystallinity)
    return tensile_strength_mpa, elastic_modulus_gpa    

def plot_pixel_to_mechanical_curve(
        laser: Laser,
        camera: Camera,
        ndfilter: NDFilter,
        ):

    """
    Plots a curve of pixel value against tensile strength and elastic modulus

    Parameters:
    - camera: Camera object
    - laser: Laser object

    Returns:
    - A plot of pixel value against tensile strength and elastic modulus
    """

    mechanical_properties = []
    pixels = list(range(camera.maximum_digital_value))
    for pixel in range(camera.maximum_digital_value):
        mechanical_properties.append(
            pixel_value_to_mechanical(pixel, laser, camera, ndfilter)
        )

    tensile_strength = [item for item, _ in mechanical_properties]
    elastic_modulus = [item for _, item in mechanical_properties]

    plt.plot(pixels, tensile_strength, label="Tensile Strength (MPa)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Tensile Strength (MPa)")
    plt.title("Pixel Value vs Tensile Strength (MPa)")
    plt.grid(True)
    plt.show()

def calculate_peak_snr(
        laser: Laser, 
        camera: Camera, 
        ndfilter: NDFilter, 
        reflectivity=0.5, 
        electrons_background=0
        ):
    """
    Calculates the SNR at the peak of the laser line considering shot noise.

    Parameters:
    - laser: Laser object
    - camera: Camera object
    - ndfilter: NDFilter object
    - reflectivity: Surface reflectivity (0-1)
    - electrons_background: Background electrons (default 0)

    Returns:
    - snr: Signal-to-noise ratio (unitless)
    - electrons_peak: Number of electrons at peak
    """
    electrons_peak = calculate_electrons_peak(laser, camera, ndfilter, reflectivity)
    noise = math.sqrt(electrons_peak + electrons_background)  # shot noise
    snr = electrons_peak / noise
    return snr, electrons_peak      

def plot_laser_line_profile(laser: Laser, camera: Camera, ndfilter: NDFilter, reflectivity=0.412, points=500):
    """
    Plots the laser line profile along the line in pixel values.
    
    Parameters:
    - laser: Laser object
    - camera: Camera object
    - ndfilter: NDFilter object
    - reflectivity: Surface reflectivity (0-1)
    - points: Number of points along the line to calculate
    """
    # Distance along the line from center
    x_positions_m = np.linspace(-laser.line_length_meters/2, laser.line_length_meters/2, points)
    sigma = laser.standard_deviation_m
    fraction = np.exp(-(x_positions_m**2)/(2*sigma**2))
    electrons_peak = calculate_electrons_peak(laser, camera, ndfilter, reflectivity)
    electrons_line = electrons_peak * fraction
    pixel_values_line = [electrons_to_digital_pixel_value(e, camera) for e in electrons_line]

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(x_positions_m*1000, pixel_values_line)  # x in mm
    plt.xlabel("Position along laser line (mm)")
    plt.ylabel("Pixel Value")
    plt.title("Laser Line Profile in Pixel Values")
    plt.grid(True)
    plt.show()


# Beam Thickness

def calculate_irradiance_through_beam(
        LED: LED,
        PEEKBeam: PEEKBeam
        ) -> float: 
    
    """
    Calculate the beam thickness at the target based on the LED parameters

    Parameters:
    - LED: LED object
    - PEEKBeam: PEEKBeam object

    Returns:
    - The amount of light passing through the PEEK beam
    """

    A = 0.15 # Absorption for PEEK at 905nm with a wall thickness of 0.5mm
    a = (A * math.log(10))/0.0005 # Absorption coefficient
    print(a)
    irradiance_through_beam = LED.irradiance/(10**((a*PEEKBeam.wall_thickness_m)/math.log(10)))

    return irradiance_through_beam

# ---------- Simulation ---------- #

# Declare Equipment Objects

# camera = Camera(
#     bit_depth=10,
#     pixel_area_m2=1.44e-10,
#     quantum_efficiency=0.5,  
#     full_well_capacity_electrons=30000,
#     frames_per_second=5        
# )

blackfly = Camera(
    bit_depth=12,
    pixel_size_m=3.45e-6,
    quantum_efficiency=0.4, # Quantum efficiency at 850nm 
    full_well_capacity_electrons=10499,
    frames_per_second=100        
)

blackfly_s = Camera(
    bit_depth=12,
    pixel_size_m=3.45e-6,
    quantum_efficiency=0.4, # Quantum efficiency at 850nm
    full_well_capacity_electrons=11170,
    frames_per_second=78
)

laser = Laser(
    output_power_watts=0.0005,
    fan_angle_degrees=20,
    distance_to_target_meters=0.03,
)

ndfilter = NDFilter(
    optical_density = 2.0,
)

# Calculate Pixel Values and Dynamic Range

plot_laser_line_profile(laser, blackfly_s, ndfilter, reflectivity=0.301)

pixel_value_maximum = calculate_peak_pixel_value(laser, blackfly_s, ndfilter, reflectivity=0.412)
pixel_value_minimum = calculate_peak_pixel_value(laser, blackfly_s, ndfilter, reflectivity=0.301)
print("Pixel Value Maximum: " + str(pixel_value_maximum))
print("Pixel Dynamic Range: " + str(pixel_value_maximum-pixel_value_minimum))

snr, electrons_peak = calculate_peak_snr(laser, blackfly, ndfilter, reflectivity=0.412)
print(f"Peak electrons: {electrons_peak:.1f}, SNR: {snr:.1f}")

peekbeam = PEEKBeam(
    wall_thickness_m=0.001,
    diameter_m=0.001,
)

led = LED(
    output_power_watts=1,
)

# output_irradiance = calculate_irradiance_through_beam(led, peekbeam)
# print("Irradiance through beam: " + str(output_irradiance) + " W/m^2")

# plot_pixel_to_mechanical_curve(laser, camera, ndfilter)