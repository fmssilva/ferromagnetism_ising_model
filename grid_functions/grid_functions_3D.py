import numpy as np

from ising_main import *
from IO_functions import *



########################## - 3D FERROMAGNETISM SIMULATIONS  - ##########################
''' Solução base para 3 dimensões, usou-se o código da solução de 2 dimensões como base alterando apenas 
    o necessãrio para aceitar os np.arrays de 3 dimensões e os ajustes ao cálculo de métricas (momento magnético, spin e energia)'''


def start_grid_with_padding(): 
    '''Start the grid with the general spin selected, 
        and with a padding in all borders to allow for easy calculation of the neighbours of a point, wihout checking boundaries''' 
    if INITIAL_GRID_SPIN == UP: 
        return np.ones((PADDED_SIDE, PADDED_SIDE, PADDED_SIDE), dtype= 'int')
    if INITIAL_GRID_SPIN == DOWN:
        return -np.ones((PADDED_SIDE, PADDED_SIDE, PADDED_SIDE), dtype= 'int')
    spin = 0.5 if INITIAL_GRID_SPIN == 0 else (INITIAL_GRID_SPIN if INITIAL_GRID_SPIN > 0 else 1 + INITIAL_GRID_SPIN) 
    return np.random.choice([UP, DOWN], (PADDED_SIDE, PADDED_SIDE, PADDED_SIDE), p=[spin, 1-spin])
    


def set_padding(grid):
    '''padding to make the grid "circular"
        only for the sides, not for the corners, as they are not used in the calculations
        and only for left and top, because inside "check spin" we already set padding for right and bottom'''
    grid[:, 0, :] = grid[:, -2, :]    # top padding == bottom column
    grid[0, :, :] = grid[-2, :, :]    # left padding == right row
    grid[:, :, 0] = grid[:, :, -2]    # front padding == back column
    return grid



def get_point_energy(grid, i , j, k, mag_field):
    '''Calculate the energy (delta) of a point in the grid
        and the energy difference if we change the spin of that point'''
    delta = grid[i][j][k] * (grid[i-1][j][k] + grid[i+1][j][k] + grid[i][j-1][k] + grid[i][j+1][k] + grid[i][j][k-1] + grid[i][j][k+1])
    energy = delta/2 - mag_field * grid[i][j][k] # /2 because for each point, we account for its energy, but also the influence it has on the neighbours
    energy_diff = 2 * delta   
    return delta, energy, energy_diff



def check_spin(grid, i, j, k, transition_values_w, mag_field):
    ''' change spin if we lose energy, so the lattice becomes more stable,
        and change also with some probability if we gain energy, to avoid local minimuns
            (we'll always change for tansition values (w) >= 1, and then random for 0 <= w < 1)
        else, keep the spin as it is
        return 1 if we change the spin, 0 otherwise, for debugging purposes'''  
    curr_spin = grid[i][j][k]
    delta , _ , energy_diff = get_point_energy(grid, i, j, k, mag_field)
    if( energy_diff < 0 or np.random.random() < transition_values_w[curr_spin][delta]): 
        grid[i][j][k] *= -1
        if(i == 1):                 # padding to make the grid "circular" with the correct values for when we get to the end of matrix 
            grid[-1][j][k] *=  -1 
        if(j == 1):
            grid[i][-1][k] *=  -1
        if(k == 1):
            grid[i][j][-1] *=  -1
        return grid, 1
    return grid, 0




def calc_cycle_metrics(grid, mag_field, spin_changes):
    '''Calculate the metrics of grid, after running a monte carlo cycle'''
         
    avg_magnetic_moment = np.sum(grid[1:-1, 1:-1, 1:-1]) / GRID_TOTAL_SIZE

    grid_energy = 0
    for i in range(1, PADDED_SIDE-1):
        for j in range(1, PADDED_SIDE-1): 
            for k in range(1, PADDED_SIDE-1):           
                grid_energy -= get_point_energy(grid, i, j, k, mag_field)[1]
                    # - energy because that represents potential energy of each point, which wants to stabelize with its neighbours in order to reduce the system's total energy    
    avg_energy = grid_energy / GRID_TOTAL_SIZE
    
    return np.array([spin_changes, avg_magnetic_moment, avg_energy])    



def monte_carlo_cycle (grid, transition_values_w, mag_field):
    '''Run a monte carlo cycle, where we check each point in the grid'''
    grid = set_padding(grid)
    spin_changes = 0
    for i in range(1, PADDED_SIDE-1):
        for j in range(1, PADDED_SIDE-1):
            for k in range(1, PADDED_SIDE-1):
                grid, spin_chang = check_spin(grid, i, j, k, transition_values_w, mag_field)
                spin_changes += spin_chang
    return grid, calc_cycle_metrics(grid, mag_field, spin_changes)


