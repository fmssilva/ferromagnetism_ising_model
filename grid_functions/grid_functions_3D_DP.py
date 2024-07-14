import math
import numpy as np
import math


from IO_functions import *
from ising_main import *
from grid_functions.grid_functions_3D import *


########################## - 3D FERROMAGNETISM SIMULATIONS  - ##########################
''' In this version with dynamic programming, we will take advantage of the monte carlo cycle 
    to calculate the metrics related to the position [i-1][j-1][k-1] 
    since the variables of all the spins that can influence it have already been calculated'''



def check_spin(grid, i, j, k, transition_values_w, mag_field):
    ''' change spin if we lose energy, so the lattice becomes more stable,
        and change also with some probability if we gain energy, to avoid local minimuns
            (we'll always change for tansition values (w) >= 1, and then random for 0 <= w < 1)
        else, keep the spin as it is
        return 1 if we change the spin, 0 otherwise, for debugging purposes'''  
    
    curr_spin = grid[i][j][k]
    delta , _ , energy_diff = get_point_energy(grid, i, j, k, mag_field)
    if( energy_diff < 0 or np.random.random() < transition_values_w[curr_spin][delta]): 
        grid[i][j][k] *=  -1
        return grid, 1
    
    return grid, 0



def _setup_grid_for_padding(grid, transition_values_w, mag_field):
    '''Calculates for values for planes x = 0, y = 0 and z = 0
    So that we build the padding without having to check for these positions that would
    change the other side of the grid''' 
    
    spin_changes = 0
    for i in range(1, PADDED_SIDE-1):
        for j in range(1, PADDED_SIDE-1):
            grid, spin_chang = check_spin(grid, i, j, 1, transition_values_w, mag_field)
            spin_changes += spin_chang
            
    for i in range(1, PADDED_SIDE-1):
        for k in range(1, PADDED_SIDE-1):
            grid, spin_chang = check_spin(grid, i, 1, k, transition_values_w, mag_field)
            spin_changes += spin_chang
    
    for j in range(1, PADDED_SIDE-1):
        for k in range(1, PADDED_SIDE-1):
            grid, spin_chang = check_spin(grid, 1, j, k, transition_values_w, mag_field)
            spin_changes += spin_chang

    return spin_chang




def _calculate_metrics_for_last_planes(grid, avg_mag_moment, avg_energy, mag_field):
    '''Calculate the metrics of the last planes of the grid, without changing the grid'''
    
    for i in range(1, PADDED_SIDE-1):
        for j in range(1, PADDED_SIDE-1):
            avg_mag_moment += grid[i-1][j-1][PADDED_SIDE-2]
            avg_energy -= get_point_energy(grid, i-1, j-1, PADDED_SIDE-2, mag_field)[1]
            
    return avg_mag_moment, avg_energy




def monte_carlo_cycle (grid, transition_values_w, mag_field):
    '''Run a monte carlo cycle, where we check each point in the grid'''
    
    spin_changes =_setup_grid_for_padding(grid, transition_values_w, mag_field)
    grid = set_padding(grid)    

    spin_changes = 0
    avg_magnetic_moment = 0
    avg_energy = 0
    
    for i in range(1, PADDED_SIDE-1):
        for j in range(1, PADDED_SIDE-1):
            for k in range(2, PADDED_SIDE-1):
                    
                grid, spin_chang = check_spin(grid, i, j, k, transition_values_w, mag_field)
                spin_changes += spin_chang
                avg_magnetic_moment += grid[i-1][j-1][k-1]
                avg_energy -= get_point_energy(grid, i-1, j-1, k-1, mag_field)[1]
                
    avg_magnetic_moment, avg_energy = _calculate_metrics_for_last_planes(grid, avg_magnetic_moment, avg_energy, mag_field)
    avg_magnetic_moment, avg_energy = _calculate_metrics_for_last_planes(grid, avg_magnetic_moment, avg_energy, mag_field)
                
    avg_magnetic_moment /= GRID_TOTAL_SIZE
    avg_energy /= GRID_TOTAL_SIZE
    
    return grid, np.array([spin_changes, avg_magnetic_moment, avg_energy])