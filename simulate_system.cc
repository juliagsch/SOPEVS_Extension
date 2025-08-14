#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include <algorithm>
#include "simulate_system.h"
#include "ev.h"
#include "params.h"
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
double sim(vector<double> &load_trace, vector<double> &solar_trace, int start_index, int end_index, double cells, double pv, double b_0, std::vector<EVRecord> evRecords, std::vector<std::vector<EVStatus>> allDailyStatuses, double max_soc, double min_soc, int Ev_start, bool writeLoad)
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
	// double sum = accumulate(solar_trace.begin(), solar_trace.end(), 0.0);
	// cout << sum << endl;
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

	int trace_length_solar = solar_trace.size();
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

				c = solar_trace[index_t_solar] * pv - hourly_load; // Remaining solar power after covering load
				d = hourly_load - solar_trace[index_t_solar] * pv; // Missing power to cover load

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

					c = solar_trace[index_t_solar] * pv - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar] * pv; // Missing power to cover load

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

					c = solar_trace[index_t_solar] * pv - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar] * pv; // Missing power to cover load

					operationResult = hybrid_unidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else if (Operation_policy == "arrival_limit")
				{
					// Check if EV should currently be charging
					if (isHome && ev_b < min_battery_charge)
					{
						maxCharging = fmin((min_battery_charge - ev_b) / (eta_c_ev * T_u), get_maxCharging(ev_b));
					}
					if (maxCharging > 0)
					{
						isCharging = true;
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] * pv - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar] * pv; // Missing power to cover load

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

					c = solar_trace[index_t_solar] * pv - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar] * pv; // Missing power to cover load

					operationResult = hybrid_bidirectional(b, ev_b, c, d, isCharging, maxCharging, isHome, hour);
				}
				else if (Operation_policy == "lbn_limit")
				{
					// Check if EV should currently be charging
					if (isHome && hour <= 4 && ev_b < min_battery_charge)
					{
						maxCharging = get_maxCharging(ev_b);
					}

					double hourly_load = load_trace[index_t_load] + maxCharging;
					total_load += hourly_load;

					c = solar_trace[index_t_solar] * pv - hourly_load; // Remaining solar power after covering load
					d = hourly_load - solar_trace[index_t_solar] * pv; // Missing power to cover load

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

// Run simulation for provides solar and load trace to find cheapest combination of
// load and solar that can meet the epsilon target
vector<SimulationResult> simulate(vector<double> &load_trace, vector<vector<double>> &solar_dataset,
								  int start_index, int end_index, double b_0, std::vector<EVRecord> evRecords, std::vector<std::vector<EVStatus>> allDailyStatuses, double max_soc, double min_soc, int Ev_start)
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
		 { return annual_sums[a] < annual_sums[b]; });
	int max_panels = solar_dataset.size();
	int trace_length = solar_dataset[0].size();
	vector<vector<double>> averaged_solar_traces;

	for (int i = 0; i < max_panels; i++)
	{
		double div = max_panels - i;
		vector<double> avg_solar_trace;
		avg_solar_trace.resize(trace_length, 0.0);
		for (int j = i; j < max_panels; j++)
		{
			int idx = sorted_indices[j];
			for (int n = 0; n < trace_length; n++)
			{
				avg_solar_trace[n] += solar_dataset[idx][n] / div / system_size_trace_pv;
			}
		}
		averaged_solar_traces.push_back(avg_solar_trace);
	}
	std::reverse(averaged_solar_traces.begin(), averaged_solar_traces.end());

	// first, find the lowest value of cells that will get us epsilon loss when the PV is maximized
	// use binary search
	double cells_U = cells_max;
	double cells_L = cells_min;

	while (cells_U - cells_L > cells_step)
	{
		double mid_cells = (cells_L + cells_U) / 2.0;
		double loss = sim(load_trace, averaged_solar_traces[0], start_index, end_index, mid_cells, pv_max, b_0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start, false);

		// cout << "sim result with " << a2_intercept << " kWh and " << pv_max << " pv: " << loss << endl;
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
	// set the starting number of battery cells to be the upper limit that was converged on
	double starting_cells = cells_U;
	double starting_cost = B_inv * starting_cells + PV_inv * pv_max;
	double lowest_feasible_pv = pv_max;

	vector<SimulationResult> curve;
	curve.emplace_back(starting_cells * kWh_in_one_cell, lowest_feasible_pv, starting_cost);
	for (double cells = starting_cells; cells <= cells_max; cells += cells_step)
	{
		// for each value of cells, find the lowest pv that meets the epsilon loss constraint
		bool binary_search = true;

		if (curve.size() >= 2)
		{
			double lastC1 = curve.end()[-1].C;
			double lastC2 = curve.end()[-2].C;

			if (lastC1 - lastC2 < 10 * pv_step)
			{
				// use linear search if last two pv values are close
				binary_search = false;
			}
		}

		if (binary_search)
		{
			double pv_U = lowest_feasible_pv;
			double pv_L = 0;

			while (pv_U - pv_L > pv_step)
			{
				double mid_pv = (pv_L + pv_U) / 2.0;
				double loss = sim(load_trace, averaged_solar_traces[ceil(mid_pv / panelkW) - 1], start_index, end_index, cells, mid_pv, b_0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start, false);

				if (loss > epsilon)
				{
					pv_L = mid_pv;
				}
				else
				{
					// (loss <= epsilon)
					pv_U = mid_pv;
				}
			}

			lowest_feasible_pv = pv_U;
		}
		else
		{
			while (true)
			{
				double loss = sim(load_trace, averaged_solar_traces[ceil((lowest_feasible_pv - pv_step) / panelkW) - 1], start_index, end_index, cells, lowest_feasible_pv - pv_step,
								  b_0, evRecords, allDailyStatuses, max_soc, min_soc, Ev_start, false);

				if (loss < epsilon)
				{
					lowest_feasible_pv -= pv_step;
				}
				else
				{
					break;
				}

				// this only happens if the trace is very short, since the battery starts half full
				// and can prevent loss without pv for a short time
				if (lowest_feasible_pv <= 0)
				{
					lowest_feasible_pv = 0;
					break;
				}
			}
		}

		double cost = B_inv * cells + PV_inv * lowest_feasible_pv;
		curve.emplace_back(cells * kWh_in_one_cell, lowest_feasible_pv, cost);
	}

	return curve;
}
