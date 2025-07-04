#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "simulate_system.h"
#include "common.h"
#include "cheby.h"
#include "ev.h"
#include <Python.h>

using namespace std;

// chunk_size: length of time (in days)
SimulationResult run_simulations(vector<double> &load, vector<vector<double>> &solar_dataset, int metric, int chunk_size, std::vector<EVRecord> evRecords, std::vector<std::vector<EVStatus>> allDailyStatuses, double max_soc, double min_soc)
{

	// set random seed to a specific value if you want consistency in results
	srand(10);

	// get number of timeslots in each chunk
	vector<vector<SimulationResult>> results;
	int t_chunk_size = chunk_size * (24 / T_u);

	// get random start times and run simulation on this chunk of data
	// compute all sizing curves
	for (int chunk_num = 0; chunk_num < number_of_chunks; chunk_num += 1)
	{

		vector<double> solar = solar_dataset[0];
		// int chunk_start = rand() % max(solar.size()%24,load.size()%24);
		int max_size = std::min(solar.size(), load.size());
		int max_chunks = max_size / 24; // Number of complete 24-hour chunks

		// Generate a random chunk index
		int chunk_index = rand() % max_chunks;

		// Calculate chunk_start
		int chunk_start = chunk_index * 24;
		int Ev_start = rand() % evRecords.size();
		int chunk_end = chunk_start + t_chunk_size;
		vector<SimulationResult> sr = simulate(load, solar_dataset, chunk_start, chunk_end, 0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start);
		// saves the sizing curve for this sample
		results.push_back(sr);
	}

#ifdef DEBUG
	// print all of the curves
	int chunk_index = 1;
	cout << "DEBUG: sizing_curves" << endl;
	for (vector<vector<SimulationResult>>::iterator it = results.begin(); it != results.end(); ++it, ++chunk_index)
	{
		cout << "chunk_" << chunk_index << endl;
		for (vector<SimulationResult>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
		{
			cout << it2->B << "\t" << it2->C << "\t" << it2->cost << endl;
		}
	}
	cout << "DEBUG: sizing_curves_end" << endl;
#endif

	// calculate the chebyshev curves, find the cheapest system along their upper envelope, and return it
	// returns the optimal result
	return calculate_sample_bound(results, epsilon, confidence);
}


// Modify the main function in your C++ file
int main(int argc, char **argv)
{
    int input_process_status = process_input(argv, true);
    if (input_process_status != 0) {
        std::cerr << "Error processing input" << std::endl;
        return 1;
    }

    // Initialize Python only once at the beginning
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/mnt/c/Users/Sharon/Desktop/SOPEVS_Single_Roof-fix_sizing')");
    
    std::string output_dir_arg = output_dir;
    std::string python_cmd = "sys.argv = ['simulation_pvlib_buildingtri_shading.py', '" + output_dir_arg + "']";
    PyRun_SimpleString(python_cmd.c_str());

    // First Python module execution
    PyObject *pModule = PyImport_ImportModule("simulation_pvlib_buildingtri_shading");
    if (pModule) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "main");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject *pResult = PyObject_CallObject(pFunc, NULL);
            if (!pResult) {
                PyErr_Print();
                std::cerr << "Python script execution failed!" << std::endl;
            }
            Py_XDECREF(pResult);
        }
        else {
            std::cerr << "Could not find main() function in Python script" << std::endl;
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        std::cerr << "Failed to load Python module" << std::endl;
        Py_Finalize();
        return 1;
    }

    std::cout << "Python script executed. Check " << output_dir << " for solar data files." << std::endl;

    // Load solar data and run simulation
    load_solar_data(output_dir);
    std::vector<EVRecord> evRecords = readEVData(path_to_ev_data);
    if (evRecords.empty()) {
        std::cerr << "Error reading EV data or no records found" << std::endl;
        Py_Finalize();
        return 1;
    }

    EVStatus evStatus;
    std::vector<std::vector<EVStatus>> allDailyStatuses = generateAllDailyStatuses(evRecords);
    SimulationResult sr = run_simulations(load, solar_dataset, metric, days_in_chunk, evRecords, allDailyStatuses, max_soc, min_soc);

    double cost = sr.B / kWh_in_one_cell * B_inv + sr.C * PV_inv;
    cout << "Battery: " << sr.B << endl << "PV: " << sr.C << endl << "Total Cost: " << cost << endl;
    
    int t_chunk_size = days_in_chunk * (24 / T_u);
    double battery_cells = sr.B / kWh_in_one_cell;
    sim(load, solar_dataset, 0, t_chunk_size, battery_cells, sr.C, 0, evRecords, allDailyStatuses, max_soc, min_soc, 0, true);

    double selected_panels = sr.C / panel_size / 0.2;
	int num_panels = static_cast<int>(std::round(selected_panels)); // Ensure integer
    cout << "number of selected panels: " << selected_panels << endl;


    // Pass the number of panels to the Python optimizer
    // Create and set Python dictionary for results to pass to solar_optimizer
    PyObject *pResultsDict = PyDict_New();
    PyDict_SetItemString(pResultsDict, "num_panels", PyLong_FromLong((long)selected_panels));
    
    // Store the dictionary to a temporary file that solar_optimizer will read
    PyObject *pSysModule = PyImport_ImportModule("sys");
    PyObject *pSysDict = PyModule_GetDict(pSysModule);
    PyObject *pPath = PyDict_GetItemString(pSysDict, "path");
    
    // Convert the dictionary to a string and write to a file
    PyRun_SimpleString("import ast");
    //std::string write_cmd = "with open('/mnt/c/Users/Sharon/Desktop/SOPEVS_Single_Roof-fix_sizing/comprehensive_results.txt', 'w') as f:\n    f.write(str(" + std::to_string(selected_panels) + "))";
    //PyRun_SimpleString(write_cmd.c_str());

	std::stringstream argv_cmd;
	argv_cmd << "import sys\n";
	argv_cmd << "sys.argv = ['solar_optimizer.py', '"
			<< num_panels << "']\n";
	PyRun_SimpleString(argv_cmd.str().c_str());

    // Second Python module execution
    PyObject *pModule_2 = PyImport_ImportModule("solar_optimizer");
    if (pModule_2) {
        PyObject *pFunc = PyObject_GetAttrString(pModule_2, "main");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject *pResult = PyObject_CallObject(pFunc, NULL);
            
            if (pResult) {
                PyObject *pTotalRadiance = PyDict_GetItemString(pResult, "total_radiance");
                PyObject *pNumPanels = PyDict_GetItemString(pResult, "num_panels");
                
                if (pTotalRadiance && pNumPanels) {
                    double total_rad = PyFloat_AsDouble(pTotalRadiance);
                    int num_panels = PyLong_AsLong(pNumPanels);
                    std::cout << "Panels selected by optimizer: " << num_panels << std::endl;
                    std::cout << "Total radiance: " << total_rad << std::endl;
                }
                Py_DECREF(pResult);
            } else {
                PyErr_Print();
            }
            Py_XDECREF(pFunc);
        }
        Py_DECREF(pModule_2);
    }
    
    // Clean up Python objects and finalize the interpreter (only once at the end)
    Py_XDECREF(pResultsDict);
    Py_XDECREF(pSysModule);
    Py_Finalize();
    
    return 0;
}