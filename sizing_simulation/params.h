// params.h
#ifndef PARAMS_H
#define PARAMS_H

#include <vector>
#include <climits>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

extern double PV_inv; // cost per unit (kW) of PV
extern vector<double> solar;

extern double B_inv; // cost per cell
extern double epsilon;
extern double confidence;
extern int metric;

extern size_t days_in_chunk;
extern size_t chunk_size;
extern size_t chunk_step;
extern size_t chunk_total;

extern vector<double> load;
extern void load_solar_data(const std::string &output_dir);
extern vector<vector<double>> solar_dataset;
extern std::string output_dir;

// define the upper and lower values to test for battery cells and pv,
// as well as the step size of the search
extern double cells_min;
extern double cells_max;
extern double cells_step; // search in step of x cells

// CONSTANTS

// defines the number of samples, set via command line input
extern size_t number_of_chunks;

/**
 * T_u: this is the time unit, representing the number of hours in
 *      each time slot of the load and solar traces
 */
constexpr size_t static T_u = 1;

/**
 * T_yr: this is year unit, representing the number of traces that constitutes a year.
 *       Inputs must have multiples of this size.
 */
const size_t static T_yr = 365 * 24 / T_u;

double static kWh_in_one_cell = 0.011284;
constexpr double static num_cells_steps = 1000; // search in total of n steps
constexpr double static num_pv_steps = 350;		// search in total of n steps

constexpr double static INFTY = numeric_limits<double>::infinity();
constexpr double static EPS = numeric_limits<double>::epsilon();

// define the upper and lower values to test for pv,
// as well as the step size of the search
extern double pv_min;
extern double pv_max;
extern double pv_step; // search in steps of x kW
extern double panelkW;
extern double system_size_trace_pv;

extern double grid_import;
extern double total_load;
extern double total_cost;
extern double total_hours;
extern double load_sum;			  // Total load used
extern double ev_power_used;	  // Total power used by ev driving (discharging to power house not included)
extern double power_lost;		  // Electricity lost due to charging and discharging efficiencies
extern double max_charging_total; // Total electricity used to charge the EV
extern double ev_battery_diff;	  // EV battery difference between beginning and start of the simulation

extern double min_soc;
extern double max_soc;
extern double ev_battery_capacity;
extern double charging_rate;
extern double discharging_rate;
extern double min_battery_charge;

extern double pv_result;
extern double battery_result;

extern std::string Operation_policy;
extern std::string path_to_ev_data;
extern int loadNumber;
extern std::string wfh_type;

struct SimulationResult
{
	double B;
	double C;
	double cost;

	SimulationResult(double B_val, double C_val, double cost_val) : B(B_val), C(C_val), cost(cost_val) {}
};

struct OperationResult
{
	double ev_b;
	double b;
	double ev_charged;
	OperationResult(double ev_b_val, double b_val, double ev_charged_val) : ev_b(ev_b_val), b(b_val), ev_charged(ev_charged_val) {}
};

extern std::vector<double> socValues;

int process_input(int argc, char **argv, bool process_metric_input);
vector<double> read_data_from_file(istream &datafile, int limit = INT_MAX);

#endif
