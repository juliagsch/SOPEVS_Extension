#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <climits>
#include <string>
#include <algorithm>
#include <set>
#include <dirent.h>

#include "common.h"

double B_inv;  // cost per cell
double PV_inv; // cost per unit (kW) of PV

double cells_min;
double cells_max;
double cells_step; // search in step of x cells
double pv_min;
double pv_max;
double pv_step; // search in steps of x kW
double battery_result;
double pv_result;
int loadNumber;

double max_soc;
double min_soc;

double ev_battery_capacity = 40.0;
int t_ch = 3;
double charging_rate = 7.4;
double discharging_rate = 7.4;

int ev_charged = 0;
int ev_discharged = 0;
int stat_charged = 0;
int stat_discharged = 0;

double grid_import = 0.0;
double total_load = 0.0;
double total_cost = 0.0;
double total_hours = 0.0;
double load_sum = 0;           // Total load used
double ev_power_used = 0;      // Total power used by ev driving (discharging to power house not included)
double power_lost = 0;         // Electricity lost due to charging and discharging efficiencies of EV
double max_charging_total = 0; // Total electricity used to charge the EV
double ev_battery_diff = 0;    // EV battery difference between beginning and start of the simulation

// common.cc
std::string EV_charging = "naive";               // Default policy
std::string Operation_policy = "unidirectional"; // Default policy
std::string path_to_ev_data;

double epsilon;
double confidence;
int metric;
int days_in_chunk;

vector<double> load;
vector<double> solar;
std::string wfh_type;

std::string output_dir;

vector<double> socValues;
vector<ChargingEvent> chargingEvents;

#include <iostream>
#include <string>
#include <cstdlib> // For std::stoi

#include <iostream>
#include <regex>
#include <string>

double panel_size;
vector<vector<double>> solar_dataset;

int extract_number(const std::string &filename);

// Helper function to extract numerical part from filenames
int extract_number(const string &filename)
{
    size_t start = filename.find_first_of("0123456789");
    if (start == string::npos)
        return -1; // No numbers found

    size_t end = filename.find_first_not_of("0123456789", start);
    if (end == string::npos)
        end = filename.length();

    return stoi(filename.substr(start, end - start));
};

// vector<double> solar;

std::string extract_wfh_type(const std::string &ev_filename)
{
    std::regex pattern("ev_data/ev_merged_T(\\d+)\\.csv");
    std::smatch match;

    if (std::regex_search(ev_filename, match, pattern) && match.size() > 1)
    {
        return match.str(1); // The captured group
    }
    else
    {
        return ""; // No match found
    }
}

/*int extractLoadNumber(const std::string &filename)
{
    // Find the position of "load_"
    size_t loadPos = filename.find("load_");
    if (loadPos == std::string::npos)
    {
        std::cerr << "Error: 'load_' not found in filename." << std::endl;
        return -1; // Error indicator
    }

    // Extract the substring starting from "load_" to the end of the filename
    std::string numberStr = filename.substr(loadPos + 5); // 5 is the length of "load_"

    // Remove ".txt" from the end of the number string
    size_t txtPos = numberStr.find(".txt");
    if (txtPos != std::string::npos)
    {
        numberStr = numberStr.substr(0, txtPos);
    }

    // Convert the number string to an integer
    int loadNumber = std::stoi(numberStr);

    return loadNumber;
}


*/

vector<double> read_data_from_file(istream &datafile, int limit = INT_MAX)
{

    vector<double> data;

    if (datafile.fail())
    {
        data.push_back(-1);
        cerr << errno << ": read data file failed." << endl;
        return data;
    }

    // read data file into vector
    string line;
    double value;

    for (int i = 0; i < limit && getline(datafile, line); ++i)
    {
        istringstream iss(line);
        iss >> value;
        data.push_back(value);
    }

    return data;
}

int process_input(char **argv, bool process_metric_input)
{

    int i = 0;

    string inv_PV_string = argv[++i];
    PV_inv = stod(inv_PV_string);

#ifdef DEBUG
    cout << "inv_PV_string = " << PV_inv
         << ", PV_inv = " << PV_inv << endl;
#endif

    string inv_B_string = argv[++i];
    B_inv = stod(inv_B_string) * kWh_in_one_cell; // convert from per-kWh to per-cell cost

#ifdef DEBUG
    cout << "inv_B_string = " << inv_B_string
         << ", B_inv = " << B_inv << endl;
#endif

    // instead of using the pv_max_string, it read the size of the panels
    // the pv_max is directly caculated according to the number of the panels and the size of panels
    string panel_width_string = argv[++i];
    double panel_width = stod(panel_width_string);
    string panel_length_string = argv[++i];
    double panel_length = stod(panel_length_string);
    panel_size = panel_width * panel_length; // Compute panel size

    // here, we assume that the max solar radiance is 1000W/m^2
    // times the size of the panel, which is in m^2. such as 1m^2 or 1.6m^2. It is decided in the solar simulation step
    // the efficiency of system is 20%
    // divided by 1000, so the unit is kW

    // original
    // string pv_max_string = argv[++i];
    // pv_max = stod(pv_max_string);

    // set default pv_min and pv_step
    pv_min = 0;
    // pv_step = (pv_max - pv_min) / num_pv_steps; // this is to assume that there are 350 panels on the rooftop. and for each rooftop it has 20/350
    pv_step = panel_size * 0.2; // pv_step = 0.2 number of panels(if we have the assumption about the area of the panels)
                                // now, I assume that the area of each panel is 1m^2.
                                // because decrease the capacity of the solar system: 1m^2 * 1000W/m^2 * 20% / 1000W = 0.2kW

    // cout << "pvmax" << pv_max << endl;

#ifdef DEBUG
    cout << "pv_max_string = " << pv_max_string
         << ", pv_max = " << pv_max
         << ", pv_min = " << pv_min
         << ", pv_step = " << pv_step
         << endl;
#endif

    string cells_max_string = argv[++i];
    cells_max = stod(cells_max_string) / kWh_in_one_cell;

    // set default cells_min and cells_step
    cells_min = 0;
    cells_step = (cells_max - cells_min) / num_cells_steps;

#ifdef DEBUG
    cout << "cells_max_string = " << cells_max_string
         << ", cells_max = " << cells_max
         << ", cells_min = " << cells_min
         << ", cells_step = " << cells_step
         << endl;
#endif

    if (process_metric_input)
    {
        string metric_string = argv[++i];
        metric = stoi(metric_string);

#ifdef DEBUG
        cout << "metric_string = " << metric_string
             << ", metric = " << metric << endl;
#endif
    }

    string epsilon_string = argv[++i];
    epsilon = stod(epsilon_string);

#ifdef DEBUG
    cout << "epsilon_string = " << epsilon_string
         << ", epsilon = " << epsilon << endl;
#endif

    string confidence_string = argv[++i];
    confidence = stod(confidence_string);

#ifdef DEBUG
    cout << "confidence_string = " << confidence_string
         << ", confidence = " << confidence << endl;
#endif

    string days_in_chunk_string = argv[++i];
    days_in_chunk = stoi(days_in_chunk_string);

#ifdef DEBUG
    cout << "days_in_chunk_string = " << days_in_chunk_string
         << ", days_in_chunk = " << days_in_chunk << endl;
#endif

    string loadfile = argv[++i];

#ifdef DEBUG
    cout << "loadfile = " << loadfile << endl;
#endif

    if (loadfile == string("--"))
    {
        // read from cin
        int limit = stoi(argv[++i]);

#ifdef DEBUG
        cout << "reading load data from stdin. limit = " << limit << endl;
#endif

        load = read_data_from_file(cin, limit);
    }
    else
    {

#ifdef DEBUG
        cout << "reading load file" << endl;
#endif

        // read in data into vector
        ifstream loadstream(loadfile.c_str());
        load = read_data_from_file(loadstream);
    }
    // loadNumber = extractLoadNumber(loadfile);

#ifdef DEBUG
    cout << "checking for errors in load file..." << endl;
#endif

    if (load[0] < 0)
    {
        cerr << "error reading load file " << loadfile << endl;
        return 1;
    }

    // Process output directory right after load file
    string output_dir_string = argv[++i];
    output_dir = output_dir_string;

    string max_soc_string = argv[++i];
    max_soc = stod(max_soc_string);

#ifdef DEBUG
    cout << "max_soc_string = " << max_soc_string << ", max_soc = " << max_soc << endl;
#endif

    string min_soc_string = argv[++i];
    min_soc = stod(min_soc_string);

#ifdef DEBUG
    cout << "min_soc_string = " << min_soc_string << ", min_soc = " << min_soc << endl;
#endif
    string ev_battery_capacity_string = argv[++i];
    ev_battery_capacity = stod(ev_battery_capacity_string);

#ifdef DEBUG
    cout << "ev_battery_capacity_string = " << ev_battery_capacity_string << ", ev_battery_capacity = " << ev_battery_capacity << endl;
#endif

    string charging_rate_string = argv[++i];
    charging_rate = stod(charging_rate_string);

#ifdef DEBUG
    cout << "charging_rate_string = " << charging_rate_string << ", charging_rate = " << charging_rate << endl;
#endif

    std::set<std::string> validOperationPolicyOptions = {"safe_arrival", "safe_departure", "bidirectional", "arrival_limit", "no_ev"};

    std::string operationPolicyInput = argv[++i];

    if (validOperationPolicyOptions.find(operationPolicyInput) == validOperationPolicyOptions.end())
    {
        std::cerr << "Invalid Operation policy: " << operationPolicyInput << std::endl;
        exit(EXIT_FAILURE);
    }

    Operation_policy = operationPolicyInput;

    string path_to_ev_data_string = argv[++i];
    path_to_ev_data = path_to_ev_data_string;
    // cout << "path_to_ev_data_string = " << path_to_ev_data_string << endl;
    wfh_type = extract_wfh_type(path_to_ev_data);
    // cout << "wfh_type = " << wfh_type << endl;

#ifdef DEBUG
    cout << " path_to_ev_data = " << path_to_ev_data << endl;
#endif

    return 0;
}

void load_solar_data(const string &output_dir)
{
    string solardir = output_dir;

    DIR *solar_dir = opendir(solardir.c_str());
    if (!solar_dir)
    {
        cerr << "Error opening solar directory: " << solardir << endl;
        exit(1);
    }

    vector<string> filenames;
    struct dirent *solar_ent;

    // Collect and sort file names
    while ((solar_ent = readdir(solar_dir)) != nullptr)
    {
        string filename = solar_ent->d_name;
        if (filename == "." || filename == ".." || filename[0] == '.')
            continue;
        filenames.push_back(filename);
    }
    closedir(solar_dir);

    sort(filenames.begin(), filenames.end(), [](const string &a, const string &b)
         { return extract_number(a) < extract_number(b); });

    // Read solar data files
    for (const string &filename : filenames)
    {
        string filepath = solardir + "/" + filename;
        ifstream filestream(filepath);
        if (!filestream)
        {
            cerr << "Error opening solar file: " << filepath << endl;
            continue;
        }

        vector<double> file_data = read_data_from_file(filestream);
        if (file_data.empty() || file_data[0] < 0)
        {
            cerr << "Error reading solar file: " << filepath << endl;
            continue;
        }

        solar_dataset.push_back(file_data);
    }

    // 计算 pv_max
    pv_max = solar_dataset.size() * panel_size * 0.2;
    cout << "panel_size: " << panel_size << endl;
    cout << "Calculated pv_max: " << pv_max << endl;
}