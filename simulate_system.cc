#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include <algorithm>
#include "simulate_system.h"
#include "ev.h"
#include "common.h"
#include <cstdlib>

using namespace std;
int loss_events = 0;
double load_deficit = 0;

// parameters specified for an NMC cell with operating range of 1 C charging and discharging
void update_parameters(double n)
{

	num_cells = n;

	a1_intercept = 0.0 * num_cells;

	a2_intercept = kWh_in_one_cell * num_cells;

	alpha_d = a2_intercept * 1.0;
	alpha_c = a2_intercept * 1.0;
	return;
}

// decrease the applied (charging) power by increments of (1/30) until the power is
// low enough to avoid violating the upper energy limit constraint.
double calc_max_charging(double power, double b_prev)
{
	double step = power / 30.0;

	for (double c = power; c >= 0; c -= step)
	{
		double upper_lim = a2_slope * (c / nominal_voltage_c) + a2_intercept;
		double b = b_prev + c * eta_c * T_u;
		if (b <= upper_lim)
		{
			return c;
		}
	}
	return 0;
}

// decrease the applied (discharging) power by increments of (1/30) until the power is
// low enough to avoid violating the lower energy limit constraint.
double calc_max_discharging(double power, double b_prev)
{

	double step = power / 30.0;

	for (double d = power; d >= 0; d -= step)
	{
		double lower_lim = a1_slope * (d / nominal_voltage_d) + a1_intercept;
		double b = b_prev - d * eta_d * T_u;
		if (b >= lower_lim)
		{
			return d;
		}
	}
	return 0;
}

double calc_max_charging_ev(double power, double ev_b)
{
	double step = power / 30.0;
	double upper_lim = max_soc * ev_battery_capacity;
	double ev_b_m = 0.0;

	for (double c = power; c > 0; c -= step)
	{
		ev_b_m = ev_b + c * eta_c_ev * T_u;
		if (ev_b_m < upper_lim)
		{
			return c;
		}
	}
	return 0;
}

double calc_max_discharging_ev(double power, double ev_b, double min_soc, double ev_battery_size)
{
	double step = power / 30.0;
	double lower_lim = min_soc * ev_battery_capacity;

	for (double d = power; d >= 0; d -= step)
	{
		// Update battery SOC
		double ev_b_m = ev_b - d * eta_d_ev * T_u;

		// Check if the updated SOC is above the lower limit
		if (ev_b_m >= lower_lim)
		{
			return d;
		}
	}
	return 0;
}

// Returns true if the EV should currently be charging, false otherwise
bool get_is_charging(const std::vector<EVStatus> &dailyStatuses, double ev_b, int currentHour)
{
	// Find the next departure hour
	int next_dept = -1;
	if (dailyStatuses[currentHour].isAtHome && ev_b < (max_soc * ev_battery_capacity - 0.1))
	{
		next_dept = convertTimeToHour(dailyStatuses[currentHour].nextDepartureTime);
	}
	if (next_dept == -1)
	{
		return false;
	}

	// Compute how many hours are needed to reach the max SOC.
	double diff = ev_battery_capacity * max_soc - ev_b;
	int hours_needed = ceil(diff / (charging_rate * eta_c_ev * T_u));

	// Find the number of hours until departure.
	int hours_until_dept = (24 + next_dept - currentHour) % 24;

	// TODO: It is possible that we don't quite reach the full battery level if we discharge in the hour before charging starts.
	if (hours_needed >= hours_until_dept)
	{
		return true;
	}
	return false;
}

OperationResult no_ev(double b, double c, double d, int hour)
{
	// There is leftover solar power after covering household and expected EV load
	if (c > 0)
	{
		double max_c = fmin(calc_max_charging(c, b), alpha_c);
		b = b + max_c * eta_c * T_u;
	}

	// Electricity is missing to cover household  load
	else if (d > 0)
	{
		double max_d = fmin(calc_max_discharging(d, b), alpha_d);
		b = b - max_d * eta_d * T_u;
		double res = d - max_d;
		if (res > 0)
		{
			loss_events += 1;
			load_deficit += res;
			grid_import += res;

			if (hour == 0 || hour == 1 || hour == 2 || hour == 3 || hour == 4)
			{
				// special grid tariff for home load
				total_cost += res * 0.07;
			}
			else
			{
				total_cost += res * 0.35;
			}
		}
	}

	return OperationResult(0, b, 0);
}

// First charge EV battery with leftover solar power and then the stationary battery
OperationResult hybrid_unidirectional(double b, double ev_b, double c, double d, bool isCharging, double maxCharging, bool isHome, int hour)
{
	double ev_charged = 0.0;
	if (isCharging == true)
	{
		ev_b = ev_b + maxCharging * eta_c_ev * T_u;
		power_lost += (maxCharging * T_u) - (maxCharging * eta_c_ev * T_u);
		// ChargingEvent event;
		// event.hour = hour;
		// event.chargingAmount = maxCharging; // TODO: Depending on use case this should be maxCharging * eta_c_ev * T_u
		// chargingEvents.push_back(event);
		ev_charged += maxCharging;
	}

	// There is leftover solar power after covering household and expected EV load
	if (c > 0)
	{
		// First we use the remaining electricity to power the EV if if is home and if it hasn't been powered already.
		if (isHome == true && isCharging == false)
		{
			double max_c_ev = calc_max_charging_ev(fmin(c, charging_rate), ev_b);
			ev_b = ev_b + max_c_ev * eta_c_ev * T_u;
			power_lost += (max_c_ev * T_u) - (max_c_ev * eta_c_ev * T_u);
			if (max_c_ev > 0)
			{
				// ChargingEvent event;
				// event.hour = hour;
				// event.chargingAmount = max_c_ev; // TODO: Depending on use case this should be max_c_ev * eta_c_ev * T_u
				// chargingEvents.push_back(event);
				ev_charged += max_c_ev;
			}
			c -= max_c_ev;
			total_load += max_c_ev;
			max_charging_total += max_c_ev;
		}
		// If there is still electricity remaining, we charge the stationary battery.
		if (c > 0)
		{
			double max_c = fmin(calc_max_charging(c, b), alpha_c);
			b = b + max_c * eta_c * T_u;
		}
	}

	// Electricity is missing to cover household and expected EV load
	else if (d > 0)
	{
		double max_d = fmin(calc_max_discharging(d, b), alpha_d);
		b = b - max_d * eta_d * T_u;
		double res = d - max_d;

		// We need to import from grid if there is still electricity missing
		if (res > 0)
		{
			loss_events += 1;
			load_deficit += res;
			grid_import += res;
			if (hour == 0 || hour == 1 || hour == 2 || hour == 3 || hour == 4)
			{
				// special grid tariff for home load
				total_cost += res * 0.07;
			}
			else
			{
				total_cost += fmin(maxCharging, res) * 0.07;
				res -= fmin(maxCharging, res);
				total_cost += res * 0.35;
			}
		}
	}
	return OperationResult(ev_b, b, ev_charged);
}

// First charge EV battery with leftover solar power and then the stationary battery
OperationResult hybrid_bidirectional(double b, double ev_b, double c, double d, bool isCharging, double maxCharging, bool isHome, int hour)
{
	double ev_charged = 0.0;
	if (isCharging == true)
	{
		ev_b = ev_b + maxCharging * eta_c_ev * T_u;
		power_lost += (maxCharging * T_u) - (maxCharging * eta_c_ev * T_u);
		// ChargingEvent event;
		// event.hour = hour;
		// event.chargingAmount = maxCharging; // TODO: Depending on use case this should be maxCharging * eta_c_ev * T_u
		// chargingEvents.push_back(event);
		ev_charged += maxCharging;
	}

	// There is leftover solar power after covering household and expected EV load
	if (c > 0)
	{
		// First we use the remaining the power the EV if if is home and if it hasn't been powered already.
		if (isHome == true && isCharging == false)
		{
			double max_c_ev = calc_max_charging_ev(fmin(c, charging_rate), ev_b);
			ev_b = ev_b + max_c_ev * eta_c_ev * T_u;
			power_lost += (max_c_ev * T_u) - (max_c_ev * eta_c_ev * T_u);
			if (max_c_ev > 0)
			{
				// ChargingEvent event;
				// event.hour = hour;
				// event.chargingAmount = max_c_ev; // TODO: Depending on use case this should be max_c_ev * eta_c_ev * T_u
				// chargingEvents.push_back(event);
				ev_charged += max_c_ev;
			}
			c -= max_c_ev;
			total_load += max_c_ev;
			max_charging_total += max_c_ev;
		}
		// If there is still electricity remaining, we charge the stationary battery.
		if (c > 0)
		{
			double max_c = fmin(calc_max_charging(c, b), alpha_c);
			b = b + max_c * eta_c * T_u;
		}
	}

	// Electricity is missing to cover household and expected EV load
	else if (d > 0)
	{
		double max_d = fmin(calc_max_discharging(d, b), alpha_d);
		b = b - max_d * eta_d * T_u;
		double res = d - max_d;

		if (res > 0 && isCharging == false && isHome == true)
		{
			double max_d_ev = fmin(calc_max_discharging_ev(res, ev_b, min_soc, ev_battery_capacity), discharging_rate / (eta_d_ev * T_u));
			ev_b = ev_b - max_d_ev * eta_d_ev * T_u;
			power_lost += (max_d_ev * eta_d_ev * T_u) - (max_d_ev * T_u);
			res = res - max_d_ev;

			// The electricity discharged from the EV needs to be excluded from the total load as it has already been included when the EV was charged.
			total_load -= max_d_ev;
		}

		// We need to import from grid if there is still electricity missing
		if (res > 0)
		{
			loss_events += 1;
			load_deficit += res;
			grid_import += res;
			if (hour == 0 || hour == 1 || hour == 2 || hour == 3 || hour == 4)
			{
				// special grid tariff for home load
				total_cost += res * 0.07;
			}
			else
			{
				total_cost += fmin(maxCharging, res) * 0.07;
				res -= fmin(maxCharging, res);
				total_cost += res * 0.35;
			}
		}
	}
	return OperationResult(ev_b, b, ev_charged);
}

double get_maxCharging(double ev_b)
{
	double available_power = charging_rate * T_u;
	return calc_max_charging_ev(available_power, ev_b);
}

double get_ev_b(std::vector<EVStatus> &dailyStatuses, int hour, double last_soc)
{
	double ev_b = last_soc;
	if (hour != 0 && !dailyStatuses[hour - 1].isAtHome && dailyStatuses[hour].isAtHome)
	{
		ev_b -= dailyStatuses[hour - 1].powerUsed;
		ev_power_used += dailyStatuses[hour - 1].powerUsed;
		// std::cout << "car arrived back home, power used: " << dailyStatuses[hour - 1].powerUsed << " last soc: " << last_soc << " new ev: " << ev_b << std::endl;
	}

	return ev_b;
}

// call it with a specific battery and PV size and want to compute the loss
double sim(vector<double> &load_trace, vector<vector<double>> &solar_trace_in, int start_index, int end_index, double cells, double b_0, std::vector<EVRecord> evRecords, std::vector<std::vector<EVStatus>> allDailyStatuses, double max_soc, double min_soc, int Ev_start, bool printGridCost)
{
	update_parameters(cells);
	loss_events = 0;
	load_deficit = 0;
	load_sum = 0;
	grid_import = 0;
	total_load = 0;
	power_lost = 0;
	total_cost = 0;
	max_charging_total = 0;
	double initial_battery_level_ev = 0;
	double last_soc = 0;

	double b = 0.0; // Start simulation with an empty stationary battery
	double c = 0.0; // Remaining solar energy after covering household and EV charging load
	double d = 0.0; // Missing energy to cover household and EV charging load after using produced solar energy.
	// We assume that the initial battery charge of the EV comes from the grid.

	// Compute aggregated solar trace
	vector<double> solar_trace;

	int trace_length_solar = 0;
	if (!solar_trace_in.empty())
	{
		int cols = solar_trace_in[0].size();
		solar_trace.resize(cols, 0.0);
		for (const auto &trace : solar_trace_in)
		{
			for (int i = 0; i < cols; ++i)
			{
				solar_trace[i] += trace[i];
			}
		}
		trace_length_solar = solar_trace.size();
	}
	else
	{
		// cerr << "Error: solar_trace_in is empty." << endl;
		return 1;
	}

	if (Operation_policy != "no_ev")
	{
		initial_battery_level_ev = fmin(32.0, ev_battery_capacity * max_soc); // EV battery level at the start of the simulation
		last_soc = initial_battery_level_ev;								  // EV battery level during the previous time step

		grid_import = initial_battery_level_ev / eta_c_ev;
		total_load = grid_import;
		total_cost = grid_import * 0.07;
		power_lost = grid_import - initial_battery_level_ev;
		max_charging_total = grid_import;
	}

	int index_t_solar; // Current index in solar trace
	int index_t_load;  // Current index in load trace

	// int trace_length_solar = solar_trace.size();
	int trace_length_load = load_trace.size();

	for (int day = 0; day < (end_index - start_index) / 24; day++)
	{
		int ev_day = day + Ev_start;
		ev_day = ev_day % 365;

		for (int hour = 0; hour < 24; hour++)
		{
			total_hours += 1;
			int t = day * 24 + start_index + hour;
			index_t_solar = t % trace_length_solar;
			index_t_load = t % trace_length_load;
			load_sum += load_trace[index_t_load];

			OperationResult operationResult = OperationResult(0, 0, 0);
			if (Operation_policy == "no_ev")
			{
				double hourly_load = load_trace[index_t_load];
				total_load += hourly_load;

				c = solar_trace[index_t_solar] - hourly_load; // Remaining solar power after covering load
				d = hourly_load - solar_trace[index_t_solar]; // Missing power to cover load

				operationResult = no_ev(b, c, d, hour);
			}
			else
			{
				bool isCharging = false;
				bool isHome = allDailyStatuses[ev_day][hour].isAtHome;
				double ev_b = get_ev_b(allDailyStatuses[ev_day], hour, last_soc);
				double maxCharging = 0.0;
				if (Operation_policy == "safe_arrival")
				{
					if (isHome == true)
					{
						maxCharging = get_maxCharging(ev_b);
					}

					if (maxCharging > 0)
					{
						isCharging = true;
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar]; // Missing power to cover load

					operationResult = hybrid_unidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else if (Operation_policy == "safe_departure")
				{
					// Check if EV should currently be charging
					isCharging = get_is_charging(allDailyStatuses[ev_day], ev_b, hour);
					if (isCharging)
					{
						maxCharging = get_maxCharging(ev_b);
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar]; // Missing power to cover load

					operationResult = hybrid_unidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else if (Operation_policy == "arrival_limit")
				{
					// Check if EV should currently be charging
					if (isHome && ev_b < ev_b * 0.2)
					{
						maxCharging = fmin((ev_b * 0.2 - ev_b) / (eta_c_ev * T_u), get_maxCharging(ev_b));
					}
					if (maxCharging > 0)
					{
						isCharging = true;
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar]; // Missing power to cover load

					operationResult = hybrid_unidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else if (Operation_policy == "bidirectional")
				{
					// Check if EV should currently be charging
					isCharging = get_is_charging(allDailyStatuses[ev_day], ev_b, hour);
					if (isCharging)
					{
						maxCharging = get_maxCharging(ev_b);
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar]; // Missing power to cover load

					operationResult = hybrid_bidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else if (Operation_policy == "lbn_limit")
				{
					// Check if EV should currently be charging
					if (isHome && hour <= 4 && ev_b < ev_b * 0.2)
					{
						maxCharging = get_maxCharging(ev_b);
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar]; // Missing power to cover load

					operationResult = hybrid_unidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else
				{
					std::cout << "ERROR: Invalid operation policy selected" << std::endl;
				}
				max_charging_total += maxCharging;

				// Update battery levels
				ev_b = operationResult.ev_b;
				last_soc = ev_b;
			}
			b = operationResult.b;
		}
	}
	ev_battery_diff = last_soc;

	if (metric == 0)
	{
		// lolp
		return loss_events / (double)(end_index - start_index);
	}
	else
	{
		// metric == 1, eue
		return load_deficit / (double)total_load;
	}
}

vector<SimulationResult> simulate(vector<double> &load_trace, vector<vector<double>> &solar_trace, int start_index, int end_index, double b_0, std::vector<EVRecord> evRecords, std::vector<std::vector<EVStatus>> allDailyStatuses, double max_soc, double min_soc, int Ev_start)
{
	// Precompute annual sums for each solar trace and sort them
	vector<double> annual_sums;
	for (int i = 0; i < solar_dataset.size(); i++)
	{
		double sum = accumulate(solar_dataset[i].begin(), solar_dataset[i].end(), 0.0);
		annual_sums.push_back(sum);
	}

	vector<size_t> sorted_indices(solar_dataset.size());
	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	sort(sorted_indices.begin(), sorted_indices.end(), [&annual_sums](size_t a, size_t b)
		 { return annual_sums[a] > annual_sums[b]; });

	// Precompute aggregated solar for all possible PV sizes (number of panels)
	int max_pv = solar_dataset.size();
	int trace_length = solar_dataset.empty() ? 0 : solar_dataset[0].size();
	double min_cost = 1000000000;
	double battery = 0;
	double pv_size = 0;
	// vector<vector<double>> aggregated_solar_all(max_pv + 1, vector<double>(trace_length, 0.0));
	// std::cout << "sorted_indices" << std::endl;
	// for (int i = 0; i < sorted_indices.size(); i++)
	// {
	// 	std::cout << sorted_indices[i] << std::endl;
	// 	std::cout << annual_sums[i] << std::endl;
	// }

	// for (int k = 1; k <= max_pv; ++k)
	// {
	// 	for (int hour = 0; hour < trace_length; ++hour)
	// 	{
	// 		double sum = 0.0;
	// 		// Use the k most productive panels (last k indices in sorted_indices)
	// 		for (int i = 0; i < k; ++i)
	// 		{
	// 			int idx = sorted_indices[max_pv - k + i];
	// 			sum += solar_dataset[idx][hour];
	// 		}
	// 		aggregated_solar_all[k][hour] = sum; // * 4 Preserve original scaling?
	// 	}
	// }

	// for (size_t idx : sorted_indices) {
	//	std::cout << idx << " ";
	// }
	//  std::cout << std::endl;

	// first, find the lowest value of cells that will get us epsilon loss when the PV is maximized
	// use binary search
	double cells_U = cells_max;
	double cells_L = cells_min;
	double mid_cells = 0.0;
	double loss = 0.0;

	// binary search to find min. battery size that works for pvmax
	while (cells_U - cells_L > cells_step)
	{

		mid_cells = (cells_L + cells_U) / 2.0;
		// simulate with PV max first
		loss = sim(load_trace, solar_trace, start_index, end_index, mid_cells, b_0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start, false);
		if (loss > epsilon)
		{
			cells_L = mid_cells;
		}
		else
		{
			// (loss <= epsilon)
			cells_U = mid_cells;
		}
	}

	// set the starting number of battery cells to the min. battery needed for pvmax
	double starting_cells = cells_U;
	double starting_cost = B_inv * starting_cells + PV_inv * pv_max;
	double lowest_feasible_pv = solar_trace.size();

	// double lowest_cost = starting_cost;
	// double lowest_B = starting_cells * kWh_in_one_cell;
	// double lowest_C = pv_max;

	vector<SimulationResult> curve;
	// first point on the curve
	curve.push_back(SimulationResult(starting_cells * kWh_in_one_cell, lowest_feasible_pv * pv_step, starting_cost));

	vector<vector<double>> optimal_solar_indices; // To store the final trace indices

	// std::cout << "First SimulationResult:" << std::endl;
	// std::cout << "  Battery (kWh): " << curve[0].B << std::endl;
	// std::cout << "  PV size: " << curve[0].C << std::endl;
	// std::cout << "  Cost: $" << curve[0].cost << std::endl;
	// cout << " steps " << cells_step << " " << pv_step << endl;

	for (double cells = starting_cells; cells <= cells_max; cells += cells_step)
	{

		// for each value of cells, find the lowest pv that meets the epsilon loss constraint
		double loss = 0;

		// vector<vector<double>> solar_subset;

		while (true) // current_pv_number > 0
		{
			// cout << lowest_feasible_pv;
			//  Least productive PVs are at the beginning (lower indices), and the most productive are at the end

			vector<vector<double>> solar_subset;
			// cout << cells * kWh_in_one_cell << " ";
			for (int i = 0; i < lowest_feasible_pv - 1; ++i)
			{
				int idx = sorted_indices[i];
				// cout << idx << " ";
				solar_subset.push_back(solar_trace[idx]);
			}
			// cout << solar_subset.size() << " ";
			// reduce pv size and find by how much you need to increase battery
			loss = sim(load_trace, solar_subset, start_index, end_index, cells, b_0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start, false);
			// cout << "loss " << loss << " ";

			if (loss < epsilon)
			{
				lowest_feasible_pv--;
				// cout << " success " << lowest_feasible_pv << endl;
			}
			else
			{
				// cout << "break " << endl;
				break;
			}

			// this only happens if the trace is very short, since the battery starts half full
			// and can prevent loss without pv for a short time
			if (lowest_feasible_pv <= 0)
			{
				// cout << "lowest possible pv set to zero " << endl;
				lowest_feasible_pv = 0;
				break;
			}
		}

		/*
		cout << "PV Size: " << current_pv_number << ", Using traces: ";
		for (int i = 0; i < current_pv_number; ++i) {
			int idx = sorted_indices[sorted_indices.size() - current_pv_number + i];
			cout << idx << " ";
		}
		cout << endl;  */

		double cost = B_inv * cells + PV_inv * lowest_feasible_pv * pv_step;
		// if (cost < min_cost)
		// {
		// 	min_cost = cost;
		// 	battery = cells * kWh_in_one_cell;
		// 	pv_size = lowest_feasible_pv * pv_step;
		// }
		// push all pv battery combinations that work
		// cout << "result " << cells * kWh_in_one_cell << " " << lowest_feasible_pv * pv_step << " " << cost << endl;
		curve.push_back(SimulationResult(cells * kWh_in_one_cell, lowest_feasible_pv * pv_step, cost));

		// if (cost < lowest_cost)
		// {
		// 	lowest_cost = cost;
		// 	lowest_B = cells * kWh_in_one_cell;
		// 	lowest_C = lowest_feasible_pv;
		// 	// optimal_solar_indices = solar_subset;
		// }
	}
	// cout << optimal_solar_indices.size() << endl;
	// cout << min_cost << " " << start_index << " " << end_index << " " << battery << " " << pv_size << endl;
	return curve;
}