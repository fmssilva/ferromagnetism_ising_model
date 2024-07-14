import numpy as np


from ising_main import *
from IO_functions import *
from grid_functions.grid_functions_3D import *




########################## - 3D FERROMAGNETISM SIMULATIONS  - ##########################



def check_spin(grid, i, j, k, transition_values_w, mag_field):
    ''' change spin if we lose energy, so the lattice becomes more stable,
        and change also with some probability if we gain energy, to avoid local minimuns
            (we'll always change for tansition values (w) >= 1, and then random for 0 <= w < 1)
        else, keep the spin as it is
        return 1 if we change the spin, 0 otherwise, for debugging purposes'''  
    delta , _ , energy_diff = get_point_energy(grid, i, j, k, mag_field)
    curr_spin = grid[i][j][k]
    if( energy_diff < 0 or np.random.random() < transition_values_w[curr_spin][delta]): 
        grid[i][j][k] *=  -1
        return grid, 1
    return grid, 0




def spin_and_padding_on_base_planes(grid, transition_values_w, mag_field):
    '''Checks the spin for the base planes, where i = 1, j = 1 or k = 1, 
        and if spin change, so we make the padding on the end edge;
        this avoids repeating multiple if checks in to check limits in a single check_spin function ''' 
    spin_changes = 0
    mag_moment = 0
    # top plane
    for i in range(1, PADDED_SIDE-1):
        for j in range(1, PADDED_SIDE-1):
            grid, spin_ch = check_spin(grid, i, j, 1, transition_values_w, mag_field)
            spin_changes += spin_ch
            mag_moment += grid[i,j,1]
            grid [i,j,-1] = grid[i,j,1]
    # left plane
    for i in range(1, PADDED_SIDE-1):
        for k in range(1, PADDED_SIDE-1):
            grid, spin_ch = check_spin(grid, i, 1, k, transition_values_w, mag_field)
            spin_changes += spin_ch
            mag_moment += grid[i,1,k]
            grid [i,-1,k] = grid[i,1,k]
    # top plane
    for j in range(1, PADDED_SIDE-1):
        for k in range(1, PADDED_SIDE-1):
            grid, spin_ch = check_spin(grid, 1, j, k, transition_values_w, mag_field)
            spin_changes += spin_ch
            mag_moment += grid[1,j,k]
            grid [-1,j,k] = grid[1,j,k]
    return grid, spin_changes, mag_moment




def monte_carlo_cycle (grid, transition_values_w, mag_field):
    '''Run a monte carlo cycle, where we check each point in the grid'''
    
    # base planes - check spin and make padding on the opposite side
    grid = set_padding(grid)  
    grid, spin_changes, mag_moment = spin_and_padding_on_base_planes(grid, transition_values_w, mag_field)

    # rest of grid - check spin and count the energy on [i-1, j-1, k-1]
    energy = 0
    for a in range(2, PADDED_SIDE-1):
        for b in range(2, PADDED_SIDE-1):
            for k in range(2, PADDED_SIDE-1):
                grid, spin_chang = check_spin(grid, a, b, k, transition_values_w, mag_field)
                spin_changes += spin_chang
                mag_moment += grid[a,b,k]
                energy -= get_point_energy(grid, a-1, b-1, k-1, mag_field)[1]

    # edge planes - count the energy on [i,j,k]
    for a in range(1, PADDED_SIDE-1):
        for b in range(1, PADDED_SIDE-1):
            energy -= get_point_energy(grid, a, b, 1, mag_field)[1]
            energy -= get_point_energy(grid, a, 1, b, mag_field)[1]
            energy -= get_point_energy(grid, 1, a, b, mag_field)[1] 
    
    # calculate the average magnetic moment and energy
    avg_magnetic_moment = mag_moment / GRID_TOTAL_SIZE
    avg_energy = energy / GRID_TOTAL_SIZE
    
    return grid, np.array([spin_changes, avg_magnetic_moment, avg_energy])