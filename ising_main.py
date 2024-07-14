
import time
import numpy as np
import multiprocessing as mp
import pyximport; pyximport.install()



''' ################################################################################## '''

'''  ##################### - SIMULATION PARAMETERS AND OPTIONS  - ##################### '''


'''  --->>> CHOOSE HERE THE TYPE OF TEST YOU WANT TO RUN AND IT'S PARAMETERS (4 steps)'''


''' Step 1: choose level of details to print during simulation '''
SHOW_CYCLE_DETAILS = False
SHOW_TEST_DETAILS = False



''' Step 2: choose the type of test and the type of code implementation to run  '''
TEST_TO_RUN = 5    # choose from 1 to 5, according to the options below
CODE_VERSION = 6   # choose from 1 to 6, according to the options below


# options of TEST_TO_RUN: 
SINGLE_TEST = 1                 #DONT CHANGE
SINGLE_MAG_FIELD_TEST = 2       #DONT CHANGE  »» TEST WITH DIFERENT INITIAL SPINS, TO SEE HOW THE RESULTS CHANGE
EVOLVING_TEMP_TEST = 3          #DONT CHANGE  »» TEST SOME TIMES TO SEE HOW, WITH LOW TEMP AND MAG FIELD = 0, THE SYSTEM EVOLVES TO BECOME A FERROMAGNET, EITHER UP OR DOWN
EVOLVING_MAG_FIELD_TEST = 4     #DONT CHANGE
ALL_TEMP_AND_MAG_FIELD_TEST = 5 #DONT CHANGE  »» TEST WITH DIFERENT INITIAL SPINS, TO SEE HOW THE RESULTS CHANGE

# options of CODE_VERSION:
MULTI_PROCESSING_2D = 1   #DONT CHANGE
MULTI_PROCESSING_3D = 2   #DONT CHANGE
DYNAMIC_PROGRAMMING_3D = 3     #DONT CHANGE
CYTHON_3D = 4                  #DONT CHANGE
CONVOLUTION_3D = 5             #DONT CHANGE
NUMBA_NJIT_3D = 6              #DONT CHANGE



''' Step 3: choose the parameters for the test you want to run '''
#  1 - SINGLE TEST values
if TEST_TO_RUN == SINGLE_TEST:
    SINGLE_TEST = True; SINGLE_MAG_FIELD_TEST = EVOLVING_TEMP_TEST = EVOLVING_MAG_FIELD_TEST = ALL_TEMP_AND_MAG_FIELD_TEST = False
    GRID_SIDE = 50
    MC_CYCLES = 20_000
    INITIAL_GRID_SPIN = -1    # -1=down, 1=up, 0=random, .decimal=random with probability of being up, -.decimal=random with probability of being down
    SINGLE_TEST_TEMPERATURE = 4.5
    SINGLE_TEST_MAGNET_FIELD = 0  
    SHOW_GRID_EVOLUTION = False     # don't choose this for big grids, it will slow the simulation
    SNAP_SHOTS_TO_PLOT = [("initial grid", 0), ("middle grid", MC_CYCLES // 2), ("75% grid", MC_CYCLES * 3 // 4), ("final grid", MC_CYCLES)]

# 2 - SINGLE_MAG_FIELD values
if TEST_TO_RUN == SINGLE_MAG_FIELD_TEST:
    SINGLE_MAG_FIELD_TEST = True; SINGLE_TEST = EVOLVING_TEMP_TEST = EVOLVING_MAG_FIELD_TEST = ALL_TEMP_AND_MAG_FIELD_TEST = False
    GRID_SIDE = 50
    MC_CYCLES = 20_000
    INITIAL_GRID_SPIN = -1    # -1=down, 1=up, 0=random, .decimal=random with probability of being up, -.decimal=random with probability of being down
    MAGN_FIELD_TEST_VALUES = np.array([0])
    TEMPERATURE_TEST_VALUES = np.arange(start=0.5, stop=5.6, step=0.1)


#  3- EVOLVING_TEMP values
if TEST_TO_RUN == EVOLVING_TEMP_TEST:
    EVOLVING_TEMP_TEST = True; SINGLE_TEST = SINGLE_MAG_FIELD_TEST = EVOLVING_MAG_FIELD_TEST = ALL_TEMP_AND_MAG_FIELD_TEST = False
    GRID_SIDE = 30
    MC_CYCLES = 2_000
    INITIAL_GRID_SPIN = -1    # -1=down, 1=up, 0=random, .decimal=random with probability of being up, -.decimal=random with probability of being down
    MAGN_FIELD_TEST_VALUES = np.array([-4., -3., -2., -1., 0. , 1., 2., 3., 4.])  
    MAGN_FIELD_TO_SHOW_ITERATIVE_EVOLUTION_OF_GRID = None # a number in MAGN_FIELD_TEST_VALUES or None to not slow the test
    # for each of the mag fields, evolve the system through this temperatures (write only the ascending values and the program will run the descending too automatically)
    TEMPERATURE_TEST_VALUES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.4, 4.5, 4.6, 5.0, 5.5, 6.0, 6.5])
    TEST_METRICS_TO_PLOT = [0,1,2,3,4] # max métrics to show = [0,1,2,3,4], corresponding to each metric of the TESTS_METRICS_NAMES variable below
    

#  4- EVOLVING_MAG_FIELD values
if TEST_TO_RUN == EVOLVING_MAG_FIELD_TEST:
    EVOLVING_MAG_FIELD_TEST = True; SINGLE_TEST = SINGLE_MAG_FIELD_TEST = EVOLVING_TEMP_TEST = ALL_TEMP_AND_MAG_FIELD_TEST = False
    GRID_SIDE = 30
    MC_CYCLES = 20_000
    INITIAL_GRID_SPIN = -1    # -1=down, 1=up, 0=random, .decimal=random with probability of being up, -.decimal=random with probability of being down
    TEMPERATURE_TEST_VALUES = np.array([0.5, 1.5, 2.5, 3.5, 4.4, 4.5, 4.6, 5.5, 6.5])
    TEMP_TO_SHOW_ITERATIVE_EVOLUTION_OF_GRID = None # a number in TEMPERATURE_TEST_VALUES or None to not slow the test
    # for each of the temperatures, evolve the system through this magnetic fields (write only the ascending values and the program will run the descending too automatically)
    MAGN_FIELD_TEST_VALUES = np.arange(start=-4, stop=4.1, step=0.25)  
    TEST_METRICS_TO_PLOT = [0,1,2,3,4] # max métrics to show = [0,1,2,3,4], corresponding to each metric of the TESTS_METRICS_NAMES variable below


#  5 - ALL_TEMP_AND_MAG_FIELD values
if TEST_TO_RUN == ALL_TEMP_AND_MAG_FIELD_TEST:
    ALL_TEMP_AND_MAG_FIELD_TEST = True; SINGLE_TEST = SINGLE_MAG_FIELD_TEST = EVOLVING_TEMP_TEST = EVOLVING_MAG_FIELD_TEST = False
    GRID_SIDE = 50
    MC_CYCLES = 20_000    # Has to be a big number to let the system reach equilibrium and measure the metrics. If too small, the results are diferent if the grid starts with all spins up or down or random
    INITIAL_GRID_SPIN = -1   # -1=down, 1=up, 0=random, .decimal=random with probability of being up, -.decimal=random with probability of being down
    TEMPERATURE_TEST_VALUES = np.array([0.5, 1.5, 2.5, 3.5, 4.4, 4.5,4.6, 5.5, 6.5]) 
    MAGN_FIELD_TEST_VALUES = np.arange(start=-4, stop=4.1, step=0.5)  



''' Step 4: run the code 
    example from the terminal run the command:     python ising_main.py
    '''



''' # --->>> END OF PARAMETERS AND OPTIONS '''





############### - CONSTANTS FOR ALL SIMULATIONS   - ###############


# --->>  Calculated automatically, don't change this values

if (SINGLE_TEST and SINGLE_TEST_TEMPERATURE <= 0) or (not SINGLE_TEST and np.any(TEMPERATURE_TEST_VALUES <= 0)):
    print("Exception!! The temperature for a single test has to be greater than 0")
    exit()


if not SINGLE_TEST:
    NUM_TEMPERATURE_VALUES = len(TEMPERATURE_TEST_VALUES)
    NUM_MAGN_FIELD_VALUES = len(MAGN_FIELD_TEST_VALUES)   
    TOTAL_TESTS = NUM_MAGN_FIELD_VALUES * NUM_TEMPERATURE_VALUES
    # Values and number of tests we are going to run

GRID_DIMENSION = 2 if CODE_VERSION == MULTI_PROCESSING_2D else 3


UP = 1
DOWN = -1
'''Values of spin'''


DELTA_OPTIONS = [-6, -4, -2, 0, 2, 4, 6]
'''Because spin has only 2 values {-1,1}:
    delta = spin(i,j,k) * Σ spin{close neighbours} => will have only these fixed values'''


CYCLES_METRICS_NAMES = ['spin_changes','Magnetic_Moment','Energy']
CYCLE_SPIN_CHANGES_IDX = 0
CYCLE_AVG_MAGNETIC_MOMENT_IDX = 1
CYCLE_ENERGY_IDX = 2
'''Metrics we are interested in for each monte carlo cycle'''


TESTS_METRICS_NAMES = ['avg Magnetic Moment', 'avg Magnetic Moment (abs)','avg Energy','Magnetic Susceptibility','Calorific Capacity']
TEST_AVG_MAGN_MOMENT_IDX = 0
TEST_AVG_MAGN_MOMENT_ABS_IDX = 1
TEST_AVG_ENERGY_IDX = 2
TEST_MGN_SUSCEPTIBILITY_IDX = 3
TEST_CALORIFIC_CAPACITY_IDX = 4
'''Metrics we are interested in'''


M_C_CYCLES_DISCARD_MARGIN = max(1, MC_CYCLES // 10 - 1)
'''Number of cycles to discard, to avoid the initial transient state of the system'''


PADDED_SIDE = GRID_SIDE + 2
'''The lattice will have a padding of 1 in each side, 
    to make it circular, without the need to check the boundaries for each grid access'''


GRID_TOTAL_SIZE = GRID_SIDE ** GRID_DIMENSION





##################### - MAIN FUNCTION  - #####################
import ising_common_function as icf
import IO_functions as io



def main():
    '''Main function, that will run the simulation according to the parameters defined above'''
    
    time_start = time.time()
    if SINGLE_TEST:
        print("Running a single test")
        temp = SINGLE_TEST_TEMPERATURE; mag_field = SINGLE_TEST_MAGNET_FIELD
        it_figure = io.start_grid_evolution_figure(temp, mag_field) if SHOW_GRID_EVOLUTION else None
        icf.calculate_ferromagnetism(temp, mag_field, it_figure, grid=None)
        time_end = time.time()
        print(f"\n\nAll tests finished in {time_end - time_start:.2f} seconds\n")
    else:  
        # prepare the multiprocessing
        # all the tests will be divided by the processes by temperatures, 
        # except for the EVOLVING_TEMP_TEST, which will be divided by magnetic fields
        # to this main values we call them tasks, and the sub values of each task we call them sub tasks
        processes = []
        tasks_queue = mp.Queue() # queue of tuple (task_value, matrix [sub_task_values x num_metrics])
        num_of_tasks = NUM_MAGN_FIELD_VALUES if EVOLVING_TEMP_TEST else NUM_TEMPERATURE_VALUES
        tasks_values = MAGN_FIELD_TEST_VALUES if EVOLVING_TEMP_TEST else TEMPERATURE_TEST_VALUES
            # if evolving temp, we will divide the tasks by magnetic fields, else by temperatures
        num_cpus = max(1, mp.cpu_count() - 2)
        num_processes = min(num_cpus, num_of_tasks)
        groups_of_tasks = np.array_split(tasks_values, num_processes)
        print(f"{num_cpus} CPUs   -   to run tasks: {groups_of_tasks}\n")
        
        # run the multiprocessing
        for process_idx, group_of_tasks_values in enumerate(groups_of_tasks):
            process = mp.Process(target=icf.run_tests_for_task_values, args=(process_idx, group_of_tasks_values, tasks_queue))
            processes.append(process)
        for process in processes:
            process.start()

        # while the processes put the results in the queue, the main process will start to process them
        # prepare matrices to organize results
        rows = NUM_MAGN_FIELD_VALUES * 2 if EVOLVING_MAG_FIELD_TEST else ( NUM_TEMPERATURE_VALUES * 2 if EVOLVING_TEMP_TEST else  NUM_MAGN_FIELD_VALUES)
        cols = NUM_MAGN_FIELD_VALUES if EVOLVING_TEMP_TEST else NUM_TEMPERATURE_VALUES
        results_per_metric = {
            metric_name: np.zeros((rows, cols), dtype=float) for metric_name in TESTS_METRICS_NAMES
        }

        # put all the tasks results from the queue in the corresponding place in the matrix [sub_task_values x task_values], and for each metric
        def process_queue():
            while not tasks_queue.empty():
                try:
                    task_value, metrics = tasks_queue.get()            
                    idx_task_value = np.where(tasks_values == task_value)[0]
                    for metric_idx, (metric_name, metric_matrix) in enumerate(results_per_metric.items()):
                        metric_matrix[:, idx_task_value] = metrics[:, metric_idx][:, np.newaxis]
                except mp.queues.Empty:
                    break

        # Periodically check the queue and process results
        while any(process.is_alive() for process in processes):
            process_queue()
            time.sleep(0.1) 
        process_queue() # Final processing of the queue after all processes have finished
        
        for process in processes:
            process.join()
        
        time_end = time.time()
        print(f"\n\nAll tests finished in {time_end - time_start:.2f} seconds\n")

        # plot results 
        if SINGLE_MAG_FIELD_TEST:
            io.plot_tests_metrics_for_a_mag_field(results_per_metric)
        elif EVOLVING_MAG_FIELD_TEST or EVOLVING_TEMP_TEST:
            io.plot_evolving_tests_metrics(results_per_metric)
        elif ALL_TEMP_AND_MAG_FIELD_TEST:   
            io.plot_tests_metrics_for_all_temp_and_mag_fields(results_per_metric)
    



if __name__ == "__main__":
    '''This code will only be executed if the script is run directly, not when it's imported'''
    main()
    