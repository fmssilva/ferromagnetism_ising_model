import numpy as np
import matplotlib.pyplot as plt

from ising_main import *


################################ - HELPER METHODS - ##################################
  
def round_matrix_values(matrix, decimals=1):
    '''Round all values in a matrix to a given number of decimals'''
    return np.round(matrix, decimals)


################################ - PRINTING METHODS - ##################################


def print_grid_metrics_during_cycles(cycle, cycle_metrics):
    '''Print the metrics of the grid in the given cycle'''
    print("-------------------")
    print(f"cycle {cycle} of {MC_CYCLES}")
    for i, metric_name in enumerate(CYCLES_METRICS_NAMES):
        print(f"{metric_name}: {cycle_metrics[i]}")



def print_grid_metrics_for_a_test(test_metrics, temp, mag_field):
    '''Print the metrics of the test after the complition of all monte carlo cycles'''
    print("\n-------------------" * 2)
    print(f"Metrics of test with   temperature: {temp}    magnetic field: {mag_field}")
    for i, metric_name in enumerate(TESTS_METRICS_NAMES):
        print(f"{metric_name}: {test_metrics[i]}")


################################ - ITERATIVE PLOTS METHODS - ##################################

# -> Iterative grid evolution plot:
def start_grid_evolution_figure(temp, mag_field):
    '''Set the figure to show the grid evolution'''
    fig, ax = plt.subplots()
    fig.suptitle(f"grid evolution with:   temperature={temp};   magnetic field={mag_field}", fontsize=12) 
    return (fig, ax) 

def update_grid_evolution_figure(it_figure, grid, temp, mag_field):
    '''Update the grid in the given axis
        if 3D grid, take the middle slice, and if the grid is too big, take the middle part of it'''
    MAX_SIZE_TO_PLOT = 100
    middle = GRID_SIDE // 2
    if grid.ndim == 3:
        grid = grid[middle, :, :]
    grid = grid[1:-1, 1:-1] # remove the padding 
    if grid.shape[0] > MAX_SIZE_TO_PLOT: # if the grid is too big, take the middle part of it
        half = MAX_SIZE_TO_PLOT // 2
        grid = grid[middle-half:middle+half, middle-half:middle+half]
    if it_figure is not None:
        it_figure[0].suptitle(f"grid evolution with:   temperature={temp};   magnetic field={mag_field}\nshowing center part of grid with {grid.shape[0]}x{grid.shape[0]}\n", fontsize=12) 
        it_figure[1].cla()  
        it_figure[1].imshow(grid, cmap='bwr', vmin=-1, vmax=1)
        plt.pause(0.01) 

def end_grid_evolution_figure(it_figure, temp, mag_field):
    '''End the grid evolution figure'''
    if it_figure is not None:
        it_figure[0].suptitle(f"end of simulation with:   temperature={temp};   magnetic field={mag_field}", fontsize=12) 
        plt.show()




################################ - STATIC PLOTS METHODS - ##################################

def plot_grid_snapshots(all_grids, temp, mag_field):
    '''Plot some grids to show the evolution of the simulation, only for 2D grids'''
    if all_grids[0].ndim != 2:
        return 
    cols = 2
    rows = (len(SNAP_SHOTS_TO_PLOT) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle(f"grid evolution snap shots for:   temperature:{temp};   magnetic field:{mag_field};   grid size:{GRID_SIDE};   total cycles:{MC_CYCLES}", fontsize=12)  
    for i, (snap_shot_name, snap_shot_idx) in enumerate(SNAP_SHOTS_TO_PLOT):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.imshow(all_grids[snap_shot_idx], cmap='bwr', vmin=-1, vmax=1)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_title(snap_shot_name)
    plt.show()



def plot_total_spin_changes_of_each_point(all_grids):
    '''Print the number of spin changes in each position of the grid during the simulation, only for 2D grids'''
    if all_grids[0].ndim != 2:
        return 
    # count spin changes in each position of the grid
    spin_change_count_grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype='int')
    for i in range(GRID_SIDE): # for each position in the grid...
        for j in range(GRID_SIDE):
            spin_change = 0
            for cycle in range(MC_CYCLES - 1): # count spin changes
                if all_grids[cycle][i, j] != all_grids[cycle + 1][i, j]:
                    spin_change += 1
            spin_change_count_grid[i, j] = spin_change
    # plot
    X, Y = np.meshgrid(np.arange(GRID_SIDE), np.arange(GRID_SIDE))  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, spin_change_count_grid, cmap='hot')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Spin Change Count')
    plt.title('Spin Change Counts in Each Position')
    plt.show()




def plot_cycles_metrics(cycles_metrics_matrix, temp, mag_field):
    '''Plot metrics of a given map of metrics (name:vector), and a main title'''
    cols = 3
    rows = (len(CYCLES_METRICS_NAMES) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4))
    fig.suptitle(f"metrics evolution during {MC_CYCLES} monte carlo cycles with:   temperature:{temp};   magnetic field:{mag_field};   grid size:{GRID_SIDE}", fontsize=12)
    for metric_idx , metric_name in enumerate(CYCLES_METRICS_NAMES):
        col = metric_idx % cols
        ax = axs[col]
        ax.plot(cycles_metrics_matrix[metric_idx], label='Metric')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel('number of cycles', fontsize=8)
        ax.set_ylabel(f"{metric_name} Value", fontsize=10)
        ax.grid(True)     
    plt.tight_layout()
    plt.show()




def plot_tests_metrics_for_a_mag_field(results_per_metric):
    '''Plot metrics vs temperature for a given magnetic field'''
    IDX_MAG_FIELD = 0
    cols = 3
    rows = (len(TESTS_METRICS_NAMES) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))  
    fig.suptitle(f"For mag_field = {MAGN_FIELD_TEST_VALUES[IDX_MAG_FIELD]}, metrics vs temperature:", fontsize=14)
    for i in range (cols * rows):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        if i < len(TESTS_METRICS_NAMES):
            metric_name = TESTS_METRICS_NAMES[i]
            metric_data = results_per_metric[metric_name].T[:,IDX_MAG_FIELD] 
            ax.plot(TEMPERATURE_TEST_VALUES, metric_data, label=metric_name)
            ax.set_title(metric_name, fontsize=10)
            ax.set_xlabel('temperature', fontsize=8)
            ax.set_ylabel(metric_name, fontsize=8)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=8) 
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()



def plot_evolving_tests_metrics(results_per_metric):
    '''Plot metrics vs evolving temperature or magnetic field for each test'''
    cols = 3
    rows = (NUM_TEMPERATURE_VALUES + cols - 1) // cols if EVOLVING_MAG_FIELD_TEST else (NUM_MAGN_FIELD_VALUES + cols - 1) // cols
    partial_title = x_label = "evolving Magnetic Field" if EVOLVING_MAG_FIELD_TEST else "evolving Temperature"
    num_plots = NUM_TEMPERATURE_VALUES if EVOLVING_MAG_FIELD_TEST else NUM_MAGN_FIELD_VALUES
    num_evolving_values = NUM_MAGN_FIELD_VALUES if EVOLVING_MAG_FIELD_TEST else NUM_TEMPERATURE_VALUES
    evolving_values = MAGN_FIELD_TEST_VALUES if EVOLVING_MAG_FIELD_TEST else TEMPERATURE_TEST_VALUES
    label_growing = "mag field growing" if EVOLVING_MAG_FIELD_TEST else "temperature growing"
    label_decaying = "mag field decaying" if EVOLVING_MAG_FIELD_TEST else "temperature decaying"
    for i, metric_name in enumerate(TESTS_METRICS_NAMES):
        if i in TEST_METRICS_TO_PLOT:
            fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
            fig.suptitle(f"{metric_name} through evolving {partial_title} with starting spin = {INITIAL_GRID_SPIN}", fontsize=12)
            for i in range(cols * rows):
                row = i // cols
                col = i % cols
                ax = axs[row, col] if rows > 1 else axs[col]
                if i < num_plots:
                    metric_data = results_per_metric[metric_name].T[i]
                    metric_data_growing = metric_data[:num_evolving_values].copy()
                    metric_data_decaying = np.flip(metric_data[num_evolving_values:].copy())
                    ax.plot(evolving_values, metric_data_growing, label=label_growing)
                    ax.plot(evolving_values, metric_data_decaying, label=label_decaying)
                    sub_title = f"with external Temperature = {TEMPERATURE_TEST_VALUES[i].round(1)}" if EVOLVING_MAG_FIELD_TEST else f"with external Magnetic Field = {MAGN_FIELD_TEST_VALUES[i].round(1)}"
                    ax.set_title(sub_title, fontsize=10, fontweight='bold')
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    ax.set_xlabel(x_label, fontsize=7)
                    ax.set_ylabel(metric_name, fontsize=9)
                    ax.legend(fontsize=6)
                    ax.grid(True)
                else:
                    ax.axis('off')
            plt.tight_layout()
            plt.show()

        

def plot_tests_metrics_for_all_temp_and_mag_fields(results_per_metric):
    '''Plot metrics vs temperature vs magnetic fields'''
    cols = 3
    rows = (len(TESTS_METRICS_NAMES) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))  
    fig.suptitle("Metrics vs Magnetic Field for each Temperature", fontsize=14)
    for i in range (cols * rows):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        if i < len(TESTS_METRICS_NAMES):
            metric_name = TESTS_METRICS_NAMES[i]
            for temp_idx, metric_values_p_temp in enumerate(results_per_metric[metric_name].T):
                ax.plot(MAGN_FIELD_TEST_VALUES, metric_values_p_temp,'-o', label=f"t: {TEMPERATURE_TEST_VALUES[temp_idx].round(1)}")
                ax.tick_params(axis='both', which='major', labelsize=6)
            ax.set_title(metric_name, fontsize=10, fontweight='bold')
            ax.set_xlabel("Magnetic Field", fontsize=8)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.legend()
            ax.grid(True)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()