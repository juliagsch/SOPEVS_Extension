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
#include "params.h"
#include "cheby.h"
#include "ev.h"

using namespace std;

// run_simulations
// load_filename: filename, each line in file contains electricity consumption value
// solar_filename: filename, each line in file contains solar generation value
// id: request id
// metric: 0 for LOLP, 1 for unmet load
// epsilon: number in range [0,1] representing LOLP or unmet load fraction.
// chunk_size: length of time (in days)
SimulationResult run_simulations(std::vector<EVRecord> evRecords, std::vector<std::vector<EVStatus>> allDailyStatuses)
{
    vector<vector<SimulationResult>> results;
    // get random start times and run simulation on this chunk of data
    for (int chunk_num = 0, chunk_start = 0;
         chunk_num < number_of_chunks; chunk_num += 1, chunk_start += chunk_step)
    {
        int Ev_start = rand() % evRecords.size();
        results.push_back(simulate(load, solar_dataset, chunk_start, chunk_start + chunk_size, 0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start));
    }

#ifdef DEBUG
    // print all of the curves
    int chunk_index = 1;
    cout << "DEBUG: sizing_curves" << endl;
    for (auto it = results.begin(); it != results.end(); ++it, ++chunk_index)
    {
        cout << "chunk_" << chunk_index << endl;
        for (auto &it2 : *it)
        {
            cout << it2.B << "\t" << it2.C << "\t" << it2.cost << endl;
        }
    }
    cout << "DEBUG: sizing_curves_end" << endl;
#endif

    // calculate the chebyshev curves, find the cheapest system along their upper envelope, and return it
    return calculate_sample_bound(results, epsilon, confidence);
}

int main(int argc, char **argv)
{
    int input_process_status = process_input(argc, argv, true);
    load_solar_data(output_dir);

    if (input_process_status)
    {
        cerr << "Illegal input" << endl;
        return 1;
    }
    // Read EV data
    std::vector<EVRecord> evRecords = readEVData(path_to_ev_data);

    // Check if EV data was read successfully
    if (evRecords.empty())
    {
        std::cerr << "Error reading EV data or no records found" << std::endl;
        return 1; // Or handle the error as appropriate
    }

    // Initialize EVStatus
    EVStatus evStatus;
    // Generate all daily statuses
    std::vector<std::vector<EVStatus>> allDailyStatuses = generateAllDailyStatuses(evRecords);

    // SimulationResult sr = run_simulations(evRecords, allDailyStatuses);
    // int Ev_start = rand() % evRecords.size();
    // double battery_cells = battery_result / kWh_in_one_cell;
    // double loss = sim(load, solar, 0, chunk_size, battery_cells, pv_result,
    // 				  0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start, false);
    // cout << loss << endl;

    SimulationResult sr = run_simulations(evRecords, allDailyStatuses);
    cout << sr.B << "\t" << sr.C << "\t" << sr.cost << endl;
    // int t_chunk_size = days_in_chunk * (24 / T_u);
    // double battery_cells = sr.B / kWh_in_one_cell;
    // sim(load, solar, 0, t_chunk_size, battery_cells, sr.C, 0, evRecords, allDailyStatuses, max_soc, min_soc, 0, true);
    return 0;
}
