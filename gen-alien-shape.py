import numpy as np
from sdf.d3 import sdf3, slab, intersection
from sdf.stl import write_binary_stl  # Import for direct STL writing

import os

from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator

# time
import time

np.random.seed(int(time.time()))

# Define the base ellipsoid SDF
@sdf3
def ellipsoid_sdf():
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        # Ellipsoid with radius 1 in XY plane, 4 along Z
        return np.sqrt((x**2 + y**2) / 1**2 + (z**2) / 6**2) - 1
    return f

# Cut away z < 0
@sdf3
def cut_ellipsoid():
    base = ellipsoid_sdf()
    def f(p):
        z = p[:, 2]
        # Use slab to keep only z >= 0
        return intersection(base, slab(z0=0))(p)
    return f

# Add 3D noise (pseudo-Perlin-like using sine waves)
def generate_noise(p, freq, amp):
    # Introduce randomness to the noise generation
    random_offset = np.random.rand(p.shape[0])  # Random values for each point, 1D array
    noise = np.sin((p[:, 0] + random_offset) * freq) * np.sin((p[:, 1] + random_offset) * freq) * np.sin((p[:, 2] + random_offset) * freq)
    return (amp * noise).reshape(-1, 1)  # Ensure it returns a 2D array with shape (n, 1)

def gen_noise_cube(octaves = 4, amp_factor = 0.5):
    amp = 1
    res = 1
    noise = np.zeros((res,res,res))
    for i in range(octaves):
        res *= 2
        amp *= amp_factor
        noise = zoom(noise, 2, order=2)
        noise += (np.random.rand(res, res, res) * 2 - 1) * amp
        print(f"Upsampled noise to {res}x{res}x{res}: {noise.min()} {noise.max()}")

    grid_coords = np.linspace(-1, 1, res)
    interpolator = RegularGridInterpolator(
        (grid_coords, grid_coords, grid_coords), 
        noise, 
        method='cubic',
        fill_value=0,
        bounds_error=False
    )

    return interpolator

@sdf3
def noisy_ellipsoid(base, octaves = 4, size = 1, center = (0,0,0), factor = 1, amp_factor = 0.5):
    interpolator = gen_noise_cube(octaves, amp_factor)
    
    def f(p):
        # transform p to the noise space
        noise_p = (p - center) / size 

        d = base(p) 

        # make linearly interpolated lookup
        interp = interpolator(noise_p).reshape(-1, 1)

        return d + interp * size * factor

    return f

@sdf3
def noise_distort(base, octaves = 4, size=1, center = (0,0,0), amplitude=1, frequency=2.0):
    x_interpolator = gen_noise_cube(octaves, amplitude)
    y_interpolator = gen_noise_cube(octaves, amplitude)
    z_interpolator = gen_noise_cube(octaves, amplitude)
    def f(p):
        noise_p = (p - center) / size 
        # distort p 
        v = np.column_stack((
            x_interpolator(noise_p),
            y_interpolator(noise_p),
            z_interpolator(noise_p)
        ))
        p += v * size 
        return base(p)
    return f


@sdf3
def sine_distort(base, size=1, amplitude=1, frequency=2.0, layers=4):
    """
    Distorts the input points using layered sine waves, inspired by Iñigo Quilez's mapP.
    - base: The base SDF function to distort.
    - amplitude: Base amplitude of the distortion (default 1.0).
    - frequency_base: Base frequency of the sine waves (default 2.0).
    - layers: Number of frequency layers (default 4).
    """
    # random xyz offset vector per layer
    offsets = np.random.rand(layers, 3) * 2 * np.pi

    def f(p):
        # Work on a copy of p to avoid modifying the input
        p_distorted = p.copy() / size

        amp = amplitude
        freq = frequency
        # Apply layered sine distortions
        for i in range(layers):
            amp *= 0.5

            # Swizzle coordinates: use yzx order for each component
            p_yzx = np.column_stack((p_distorted[:, 1], p_distorted[:, 2], p_distorted[:, 0])) 

            # apply offset
            p_yzx += offsets[i]

            distortion = amp * np.sin(freq * p_yzx)
            p_distorted += distortion

            freq *= 2
        
        # Evaluate the base SDF on the distorted points
        return base(p_distorted * size)
    
    return f

@sdf3
def twist(base, angle_per_z=np.pi/6):  # default to 30 degrees per unit z
    """
    Applies a twist deformation around the z-axis.
    - base: The base SDF to twist
    - angle_per_z: Rotation angle per unit of z height (in radians)
    """
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        
        # Calculate rotation angle based on z coordinate
        angle = z * angle_per_z
        
        # Create rotation matrix elements
        c = np.cos(angle)
        s = np.sin(angle)
        
        # Apply rotation around z-axis
        x_new = x * c - y * s
        y_new = x * s + y * c
        
        # Construct twisted points
        p_twisted = np.column_stack((x_new, y_new, z))
        
        return base(p_twisted)
    return f

# Generate and save the mesh
def main():
    base_sdf = cut_ellipsoid()
    noised_sdf = noisy_ellipsoid(base_sdf, 4, 7, (0,0,3), 0.5, 0.3)
    
    # either sine or noise distortion
    # distorted_sdf = noise_distort(noised_sdf, 5, 8, (0,0,3), 0.4, 3)
    distorted_sdf = sine_distort(noised_sdf, 10, 0.5, 5, 6)

    # Add twist transformation before the noise distortion
    random_angle = (np.random.rand() + 1 * np.pi) / 6
    twisted_sdf = twist(distorted_sdf, angle_per_z=random_angle)  # 45 degrees per unit z
    
    # High resolution for detail and smoothness
    points = twisted_sdf.generate(
        samples=2**24,  # Lots of samples for a detailed mesh
        step=0.01,     # Small step size for smoothness
        verbose=True   # Print progress
    )
    
    postfix = ''
    outfile = lambda: f'results-twisted/noisy_ellipsoid{postfix}.stl'
    i = 1 
    while os.path.exists(outfile()):
        i += 1
        postfix = f'_{i}'

    outfile = outfile()

    # Save directly as STL using write_binary_stl
    write_binary_stl(outfile, points)
    print(f"STL file saved as '{outfile}'")

if __name__ == "__main__":
    main()