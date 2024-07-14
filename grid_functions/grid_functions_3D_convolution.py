import numpy as np


from IO_functions import *
from ising_main import *

from scipy.ndimage import convolve
from grid_functions.grid_functions_3D import *

########################## - 3D FERROMAGNETISM SIMULATIONS  - ##########################
''' Solution for 3 dimensions, using convolution'''


CHECK_SPIN_RANDOM_STEP = 6
CHECK_SPIN_RANDOM_STEP_PROB = 1/CHECK_SPIN_RANDOM_STEP
'''The number of neighbours of a point in the grid, for us to give some independence of the spin changes between points'''


'''The kernel is a 3x3x3 matrix, that will be used to calculate the energy of each point in the grid'''
kernel = np.array([[[0, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 0]],

                   [[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]],

                   [[0, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 0]]])



def get_energy_grids(grid, mag_field):
    '''Calculate the delta and the energy of each point in the grid, and the energy difference if we change the spin of that point'''
    neighbours = convolve(grid, kernel, mode='wrap') 
    delta_grid = grid * neighbours                   
    energy_grid = delta_grid/2 - mag_field * grid
    energy_diff_grid = 2 * delta_grid
    return delta_grid, energy_grid, energy_diff_grid



def check_spin(grid, delta_grid, transition_values_w, energy_diff_grid):
    ''' change spin if we lose energy, so the lattice becomes more stable,
        and change also with some probability if we gain energy, to avoid local minimuns'''
    transition_probs = transition_values_w[grid.flatten(), delta_grid.flatten()].reshape(grid.shape)
    random_grid = np.random.random(grid.shape)
    spin_changes_grid =  (random_grid <= CHECK_SPIN_RANDOM_STEP_PROB) & ( (energy_diff_grid < 0) | (np.random.random(grid.shape) < transition_probs))

    grid[spin_changes_grid] *= -1

    return grid, np.sum(spin_changes_grid)




def cal_metrics(grid, mag_field, tot_spin_changes):
    '''Calculate the metrics of grid, after running a monte carlo cycle'''

    avg_magnetic_moment = np.mean(grid[1:-1, 1:-1, 1:-1])

    _, energy_grid, _ = get_energy_grids(grid, mag_field)
    avg_energy = -np.mean(energy_grid[1:-1, 1:-1, 1:-1])

    return np.array([tot_spin_changes, avg_magnetic_moment, avg_energy])




def monte_carlo_cycle (grid, transition_values_w, mag_field):
    '''Run a monte carlo cycle, where we check each point in the grid'''
    
    tot_spin_changes = 0
    for _ in range(CHECK_SPIN_RANDOM_STEP):
        delta_grid, _, energy_diff_grid = get_energy_grids(grid, mag_field)
        grid, tot_spin_ch = check_spin(grid, delta_grid, transition_values_w, energy_diff_grid)
        tot_spin_changes += tot_spin_ch
    
    return grid, cal_metrics(grid, mag_field, tot_spin_changes)
    

