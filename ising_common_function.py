import math
import importlib
import numpy as np
import IO_functions as io

# importing our global variables
from ising_main import *



if CODE_VERSION == MULTI_PROCESSING_2D:
    fun = importlib.import_module('grid_functions.grid_functions_2D')
elif CODE_VERSION == MULTI_PROCESSING_3D:
    fun = importlib.import_module('grid_functions.grid_functions_3D')
elif CODE_VERSION == DYNAMIC_PROGRAMMING_3D:
    fun = importlib.import_module('grid_functions.grid_functions_3D_DP')
elif CODE_VERSION == CONVOLUTION_3D:
    fun = importlib.import_module('grid_functions.grid_functions_3D_convolution')
elif CODE_VERSION == CYTHON_3D:    
    import grid_functions.grid_functions_3D_Cython as fun # type: ignore otherwise it would give a warning since it doesn't detect the cython file
elif CODE_VERSION == NUMBA_NJIT_3D: 
    fun = importlib.import_module('grid_functions.grid_functions_3D_Numba') 



def get_transition_values_w(temp, mag_field):
    '''Because the values of temperature and magnetic_field are constants throughout each simulation
        we can pre calculate the transition values (w) for each spin and delta, beeing: 
            w = e ^ ( - 2 (delta + spin * magnetic_field) / temperature)
        And so if (d_sm = (delta + spin * magnetic_field) <= 0), will have w = e ^ d_sm => w > 1
            and because we'll use a random number between 0 and 1 to decide if we change the spin,
            we can avoid the calculation of w with exponential and division, and just store 1 directly, 
            and we'll always change spin for tansition values (w) >= 1
        And if d_sm > 0, will have w = 1 / (e ^ d_sm/t ) => 0 <= w < 1, so for these cases we calculate the precise value of w '''
    delta_length = max(DELTA_OPTIONS) + abs(min(DELTA_OPTIONS)) + 1
    spin_length = 3 
    transition_values_spin = np.zeros((spin_length, delta_length))
    for spin in [UP, DOWN]:
        for idx, delta in enumerate(DELTA_OPTIONS):
            d_sm = delta + (spin * mag_field)
            if d_sm <= 0:
                transition_values_spin[spin, delta] = 1
            else:
                transition_values_spin[spin, delta] = math.exp(-(2*d_sm) / temp)
    return transition_values_spin



def calc_test_metrics(temp, mag_field, cycles_metrics_matrix):
    '''Calculate the metrics of a test, discarding the first 10% of the cycles to avoid the initial unstable state of the grid'''
    filtered_cycles_avg_mag_moment = cycles_metrics_matrix[CYCLE_AVG_MAGNETIC_MOMENT_IDX][M_C_CYCLES_DISCARD_MARGIN: ]
    filtered_cycles_avg_energy = cycles_metrics_matrix[CYCLE_ENERGY_IDX][M_C_CYCLES_DISCARD_MARGIN: ]

    test_avg_mag_moment = np.mean(filtered_cycles_avg_mag_moment)

    test_avg_mag_moment_abs = np.mean(np.abs(filtered_cycles_avg_mag_moment))

    test_avg_energy = np.mean(filtered_cycles_avg_energy)

    test_mag_susc = np.var(filtered_cycles_avg_mag_moment) * GRID_TOTAL_SIZE / temp
    
    test_calorific_cap = np.var(filtered_cycles_avg_energy) / (GRID_TOTAL_SIZE * temp**2)
    
    test_metrics = np.array([test_avg_mag_moment, test_avg_mag_moment_abs, test_avg_energy, test_mag_susc, test_calorific_cap])
    if SHOW_CYCLE_DETAILS or SINGLE_TEST:
        io.print_grid_metrics_for_a_test(test_metrics, temp, mag_field)
    
    return test_metrics
  




def calculate_ferromagnetism(temp, mag_field = 0, it_figure = None, grid = None):
    '''Run the simulation of ferromagnetism for a given temperature and magnetic field values'''    
    if grid is None:
        grid = fun.start_grid_with_padding() 
    transition_values_w = get_transition_values_w(temp, mag_field)
    all_cycles_metrics_matrix = np.zeros((len (CYCLES_METRICS_NAMES), MC_CYCLES), dtype=float)

    if SHOW_CYCLE_DETAILS or SINGLE_TEST:
        io.update_grid_evolution_figure(it_figure, grid, temp, mag_field)  
        
        # save all grids for later analysis
        def save_grid_snap (grid):
            if( GRID_DIMENSION == 2):
                return grid[1:-1, 1:-1].copy()
            middle = GRID_SIDE // 2  # elif 3D - take a slice of the grid in the middle
            return grid[middle, 1:-1, 1:-1].copy()
        all_grids = np.zeros((MC_CYCLES + 1, GRID_SIDE, GRID_SIDE), dtype='int')
        all_grids[0] = save_grid_snap (grid)

        time_start = time.time()
        for cycle in range(MC_CYCLES):
            print(f"Cycle {cycle + 1} of {MC_CYCLES}")
            grid, cycle_metrics = fun.monte_carlo_cycle(grid, transition_values_w, mag_field)
            all_cycles_metrics_matrix[:, cycle] = cycle_metrics
            if SHOW_CYCLE_DETAILS:
                io.print_grid_metrics_during_cycles(cycle, io.round_matrix_values(cycle_metrics))
            io.update_grid_evolution_figure(it_figure, grid, temp, mag_field)
            all_grids[cycle + 1] = save_grid_snap (grid)
        time_end = time.time()
        print(f"Time to run {MC_CYCLES} cycles: {time_end - time_start} seconds")
        
        if SINGLE_TEST:
            io.end_grid_evolution_figure(it_figure, temp, mag_field)
            io.plot_grid_snapshots(all_grids, temp, mag_field)
            io.plot_total_spin_changes_of_each_point(all_grids)
            io.plot_cycles_metrics(all_cycles_metrics_matrix, temp, mag_field)
    
    elif it_figure is not None:
        for cycle in range(MC_CYCLES):
            io.update_grid_evolution_figure(it_figure, grid, temp, mag_field)
            grid, cycle_metrics = fun.monte_carlo_cycle(grid, transition_values_w, mag_field)
            all_cycles_metrics_matrix[:, cycle] = cycle_metrics
    
    else: # run simple monte carlo cycles 
        for cycle in range(MC_CYCLES):
            grid, cycle_metrics = fun.monte_carlo_cycle(grid, transition_values_w, mag_field)
            all_cycles_metrics_matrix[:, cycle] = cycle_metrics

    return grid, calc_test_metrics(temp, mag_field, all_cycles_metrics_matrix)




def run_tests_for_task_values(process_idx, task_values, metrics_queue):
    '''Run the tests for the given task-values (which might be temperatures or magnetic field values, depending on the type of test),
        and store the metrics in a multi processing queue to be processed later'''

    print(f"Process {process_idx} is starting tests for task values = {io.round_matrix_values(task_values)}")
    for task_idx, task_value in enumerate(task_values):
        # if evolving temperature or mag fields we will increase the values step by step, and then decrease, 
        # and also in this cases we'll be using the previous grid to continue the test evolving from there, thus this set of configurations to lunch the tests
        rows = NUM_MAGN_FIELD_VALUES * 2 if EVOLVING_MAG_FIELD_TEST else ( NUM_TEMPERATURE_VALUES * 2 if EVOLVING_TEMP_TEST else NUM_MAGN_FIELD_VALUES)
        cols = len (TESTS_METRICS_NAMES)
        tests_metrics_for_a_task_value = np.zeros((rows, cols), dtype=float)
        grid = fun.start_grid_with_padding() if (EVOLVING_MAG_FIELD_TEST or EVOLVING_TEMP_TEST) else None
        sub_tasks_values = TEMPERATURE_TEST_VALUES if EVOLVING_TEMP_TEST else MAGN_FIELD_TEST_VALUES          
        evolve_tests_times = 2 if (EVOLVING_MAG_FIELD_TEST or EVOLVING_TEMP_TEST) else 1 # if we are evolving the magnetic field or temperature, we will run the tests twice, first increasing the values, and then decreasing
        totSubTasks = len(sub_tasks_values) if evolve_tests_times == 1 else len(sub_tasks_values) * 2

        it_figure = None
        if EVOLVING_MAG_FIELD_TEST and task_value == TEMP_TO_SHOW_ITERATIVE_EVOLUTION_OF_GRID:
            it_figure = io.start_grid_evolution_figure(task_value, sub_tasks_values[0]) 
        elif EVOLVING_TEMP_TEST and task_value == MAGN_FIELD_TO_SHOW_ITERATIVE_EVOLUTION_OF_GRID:
            it_figure = io.start_grid_evolution_figure(sub_tasks_values[0], task_value)
        
        for evolve_times in range(evolve_tests_times):
            if evolve_times == 1: 
                sub_tasks_values = np.flip(sub_tasks_values) # if evolving second time, we will reverse the order of the magnetic fields or temperatures to decrease the values, and remove the last one to not repeat
            for i, sub_task_value in enumerate (sub_tasks_values):
                print(f"Process {process_idx} is starting: sub_task {i + 1} of {totSubTasks} - of task {task_idx + 1} of {len(task_values)}")
                grid = grid if (EVOLVING_MAG_FIELD_TEST or EVOLVING_TEMP_TEST) else None
                if EVOLVING_TEMP_TEST:
                    grid, test_metrics = calculate_ferromagnetism(sub_task_value, task_value, it_figure, grid)
                else:
                    grid, test_metrics = calculate_ferromagnetism(task_value, sub_task_value, it_figure, grid)
                idx_sub_task_value = np.where(sub_tasks_values == sub_task_value)[0] 
                if evolve_times == 1:
                    idx_sub_task_value = NUM_MAGN_FIELD_VALUES + idx_sub_task_value if EVOLVING_MAG_FIELD_TEST else NUM_TEMPERATURE_VALUES + idx_sub_task_value
                tests_metrics_for_a_task_value[idx_sub_task_value] = test_metrics
        if SHOW_TEST_DETAILS:
            print(io.round_matrix_values(tests_metrics_for_a_task_value))
        if EVOLVING_MAG_FIELD_TEST and task_value == TEMP_TO_SHOW_ITERATIVE_EVOLUTION_OF_GRID:
            io.end_grid_evolution_figure(it_figure, task_value, sub_tasks_values[0])
        if EVOLVING_TEMP_TEST and task_value == MAGN_FIELD_TO_SHOW_ITERATIVE_EVOLUTION_OF_GRID:
            io.end_grid_evolution_figure(it_figure, sub_tasks_values[0], task_value)
        
        # save the results for this task, with the values of all the metrics for all the sub task values
        metrics_queue.put((task_value, tests_metrics_for_a_task_value.copy())) 

