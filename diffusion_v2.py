import numpy as np
import matplotlib.pyplot as plt

# Define the lattice size and time steps
nrows = 50
ncols = 50
num_steps = 10000
dt = 0.0001
dx = 0.01
dy = 0.01
alpha = 0.1

# Initialize the lattice with zeros
lattice = np.zeros((nrows, ncols))

# Set the temperature of some squares on the lattice
lattice[20:30, 20:30] = 10  # Set a square region to 100 degrees

# Set the boundary conditions to zero temperature
lattice[:,0] = 0
lattice[:,-1] = 0
lattice[0,:] = 0
lattice[-1,:] = 0

# Run the simulation
for step in range(num_steps):
    # Copy the current lattice to a new lattice for the next time step
    new_lattice = lattice.copy()

    # Update the temperature of each node based on the 2D diffusion equation
    for i in range(1, nrows-1):
        for j in range(1, ncols-1):
            new_lattice[i,j] = (lattice[i,j] + alpha * dt * 
                                ((lattice[i+1,j] - 2*lattice[i,j] + lattice[i-1,j])/(dx*dx) +
                                (lattice[i,j+1] - 2*lattice[i,j] + lattice[i,j-1])/(dy*dy)))

    # Set the boundary conditions to zero temperature
    new_lattice[:,0] = 0
    new_lattice[:,-1] = 0
    new_lattice[0,:] = 0
    new_lattice[-1,:] = 0

    # Update the lattice for the next time step
    lattice = new_lattice

# Plot the final temperature distribution
plt.imshow(lattice, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
