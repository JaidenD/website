import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
import os
import webbrowser
import imageio

# Define the lattice size and time steps
nrows = 50
ncols = 50
num_steps = 10000
dt = 0.0001
dx = 0.01
dy = 0.01
alpha = 0.1
threshold = 1e-5

# Initialize the lattice with zeros
lattice = np.zeros((nrows, ncols))

# Set the temperature of some squares on the lattice
lattice[20:30, 20:30] = 100

# Set the boundary conditions to zero temperature
lattice[:, 0] = 0
lattice[:, -1] = 0
lattice[0, :] = 0
lattice[-1, :] = 0

# Specify where to save individual frames to
frames_folder = "frames_folder = diffusion_frames"
os.makedirs(frames_folder, exist_ok=True)

# Prepare for creating a gif
frames = []
gif_step = 50
step = 0

# Set the color scale limits
vmin = 0
vmax = 10
while True:
    # Calculate the new lattice using vectorized operations
    new_lattice = lattice.copy()
    new_lattice[1:-1, 1:-1] = lattice[1:-1, 1:-1] + alpha * dt * (
        (lattice[2:, 1:-1] - 2 * lattice[1:-1, 1:-1] + lattice[:-2, 1:-1]) / (dx * dx)
        + (lattice[1:-1, 2:] - 2 * lattice[1:-1, 1:-1] + lattice[1:-1, :-2]) / (dy * dy)
    )

    # Set the boundary conditions to zero temperature
    new_lattice[:, 0] = 0
    new_lattice[:, -1] = 0
    new_lattice[0, :] = 0
    new_lattice[-1, :] = 0

    frames_folder = "C:\\Users\\jd21az\\OneDrive - Brock University\\Desktop\\diffusuion frames"
    os.makedirs(frames_folder, exist_ok=True)

    # Calculate the maximum change in temperature
    max_change = np.max(np.abs(new_lattice - lattice))

    # Update the lattice for the next time step
    lattice = new_lattice

    # Save the frame for the gif every gif_step steps
    if step % gif_step == 0:
        plt.imshow(lattice, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'Step {step}')
        plt.savefig('frame.png')
        plt.savefig(os.path.join(frames_folder, f'frame_{step}.png'))
        plt.clf()
        frames.append(iio.imread('frame.png'))

    step += 1

    # Break the loop when the maximum temperature change is below the threshold
    if max_change < threshold:
        break

# Save the gif
imageio.mimsave('heat_diffusion.gif', frames, 'GIF', duration=0.2)

# Automatically open the gif
webbrowser.open('heat_diffusion.gif')

# Print the message with the directory where the images are saved
print(f"Images saved to the following directory: {frames_folder}")
