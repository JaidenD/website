import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
import os
import webbrowser
import imageio

#README
#To anyone reading the code: Remember to install the relevant python libraries (above) or else the code won't run

#This program models heat diffusion on a lattice using the finite difference method to approximate the 2D heat equations
#This program outputs 2 things:
#A gif and a folder of images taken every 50 steps
#Once done processing (may take 1 min+ depending on your performance) the gif should open in your browser (if it doesnt it may be a permission issue)
#Additionally printed in the terminal is the directory where the frames folder is saved
#This program also shows a plot of the average tempreature plotted against steps, and the plot of the rate of change of average temprature

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

# Specify where to save individual frames to (user's desktop)
desktop_path = os.path.expanduser("~/Desktop")
frames_folder = os.path.join(desktop_path, "diffusion_frames")
os.makedirs(frames_folder, exist_ok=True)

# Prepare for creating a gif and storing average temperatures
frames = []
gif_step = 50
step = 0
average_temperatures = []

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

    os.makedirs(frames_folder, exist_ok=True)

    # Calculate the maximum change in temperature
    max_change = np.max(np.abs(new_lattice - lattice))

    # Update the lattice for the next time step
    lattice = new_lattice

    # Save the frame for the gif every gif_step steps and calculate the average temperature
    if step % gif_step == 0:
        plt.imshow(lattice, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'Step {step}')
        plt.savefig('frame.png')
        plt.savefig(os.path.join(frames_folder, f'frame_{step}.png'))
        frames.append(iio.imread('frame.png'))
        plt.close()
        
        # Calculate and store the average temperature
        average_temperature = np.mean(lattice[1:-1, 1:-1])
        average_temperatures.append(average_temperature)

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

# Plot the average temperature every 50 steps
plt.figure()
plt.plot(range(0, step, gif_step), average_temperatures)
plt.xlabel('Step')
plt.ylabel('Average Temperature')
plt.title('Average Temperature vs. Step')
plt.savefig(os.path.join(frames_folder, 'average_temperature.png'))
plt.show(block=False)

# Calculate the derivative of median temperatures
median_temperature_derivative = np.diff(average_temperatures)

# Find the index of the maximum absolute rate of change in median temperature
max_abs_derivative_index = np.argmax(np.abs(median_temperature_derivative))

# Find the step at which the maximum absolute rate of change occurs
max_derivative_step = (max_abs_derivative_index + 1) * gif_step

# Print the step of the maximum absolute rate of change
print(f"Step with the maximum absolute rate of change in median temperature: {max_derivative_step}")



# Plot the derivative of median temperature every 50 steps
plt.figure()
plt.plot(range(gif_step, step, gif_step), median_temperature_derivative)
plt.xlabel('Step')
plt.ylabel('Derivative of Median Temperature')
plt.title('Derivative of Median Temperature vs. Step')
plt.savefig(os.path.join(frames_folder, 'median_temperature_derivative.png'))
plt.show()

