""" 
Converts the raw daily load traces from Faraday to annual traces of each individual household.
Contains functions to plot the load traces.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def process_responses(input_file):
    """
    Appends the daily load profiles of a population containing x households to x seperate files containing the load traces of the households individually.
    By iterating through all of the daily load profiles of a year, the yearly load traces of the households are created.

    Args:
        input_file: daily load profile of x households
    """

    with open(input_file, 'r') as file:
        data = json.load(file)

    # Mapping from Faraday population name to archetype.
    population = {
        # "NoLCT": "NoLCT",
        # "EV": "EV",
        "DetachedA": "Detached",
        "DetachedD": "Detached",
        "TerracedA": "Terraced",
        "TerracedD": "Terraced",
        "Semi-detachedA": "Semi-detached",
        "Semi-detachedD": "Semi-detached"
    }

    for p in population:
        os.makedirs(f"./policy_data/faraday_processed/{population[p]}", exist_ok=True)

        # Extract relevant results
        response = next((item for item in data["message"]["results"] if item["name"] == p), None)
        kwh = response["kwh"]

        # Write to output text file with each hourly value on a new line
        for idx, daily_trace in enumerate(kwh):
            with open(f'./policy_data/faraday_processed/{population[p]}/{p}_{idx}.txt', 'a') as file:
                for i in range(0, len(daily_trace) - 1, 2):
                    # Get hourly values by adding half-hourly load
                    file.write(f"{float(daily_trace[i]) + float(daily_trace[i + 1])}\n")


def get_hourly_average(file_path):
    """
    Calculate the hourly load average of a household. 

    Args:
        file_path: Path to load trace of a household.
    
    Returns:
        hourly_agerages: Average load during each hour of the day.
    """
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    
    # Calculate average hourly production
    hourly_averages = [np.mean(data[hour::24]) for hour in range(24)]
    
    return hourly_averages


def get_monthly_load(file_path):
    """
    Calculate the average daily load for each month.

    Args:
        file_path: Path to annual load trace of a household (should contain 365 days).
    
    Returns:
        monthly_load: Average daily load for each month of the year.
    """
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    
    # Assume data has 365 days, so we split it into 12 months
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    monthly_load = []
    start_idx = 0
    for days in days_in_month:
        end_idx = start_idx + days
        monthly_load.append(np.sum(data[start_idx * 24:end_idx * 24])/float(days))
        start_idx = end_idx

    return monthly_load

def analyze_and_plot_hourly(load_files):
    """
    Plots the average load over a day for each archetype.

    Args:
        load_files: File paths to load traces of x households which should be included.
    """
    total_detached = [0.0 for _ in range(24)]
    total_semi_detached = [0.0 for _ in range(24)]
    total_terraced = [0.0 for _ in range(24)]
    detached_count, semi_detached_count, terraced_count = 0.0,0.0,0.0

    for load_file in load_files:
        hourly = get_hourly_average(load_file)
        if "Semi-detached" in load_file:
            total_semi_detached = np.add(total_semi_detached, hourly)
            semi_detached_count+=1
        elif "Detached" in load_file:
            total_detached = np.add(total_detached, hourly)
            detached_count+=1
        elif "Terraced" in load_file:
            total_terraced = np.add(total_terraced, hourly)
            terraced_count+=1

    plt.figure(figsize=(10, 5))

    plt.plot(range(24), [total/detached_count for total in total_detached], label="Detached", marker='o')
    plt.plot(range(24), [total/semi_detached_count for total in total_semi_detached], label="Semi-Detached", marker='o')
    plt.plot(range(24), [total/terraced_count for total in total_terraced], label="Terraced", marker='o')

    plt.title('Average Hourly Load')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Load (kWh)')
    plt.xticks(range(24), labels=[f'{hour}:00' for hour in range(24)], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./policy_data/faraday_processed/average_per_hour.png')
    plt.show()

def analyze_and_plot_monthly(load_files):
    """
    Plots the monthly load for each archetype.

    Args:
        load_files: File paths to load traces of x households which should be included.
    """
    total_detached = [0.0 for _ in range(12)]
    total_semi_detached = [0.0 for _ in range(12)]
    total_terraced = [0.0 for _ in range(12)]
    detached_count, semi_detached_count, terraced_count = 0.0,0.0,0.0

    for load_file in load_files:
        hourly = get_monthly_load(load_file)
        if "Semi-detached" in load_file:
            total_semi_detached = np.add(total_semi_detached, hourly)
            semi_detached_count+=1
        elif "Detached" in load_file:
            total_detached = np.add(total_detached, hourly)
            detached_count+=1
        elif "Terraced" in load_file:
            total_terraced = np.add(total_terraced, hourly)
            terraced_count+=1

    plt.figure(figsize=(12, 6))

    plt.plot(range(12), [total/detached_count for total in total_detached], label="Detached", marker='o')
    plt.plot(range(12), [total/semi_detached_count for total in total_semi_detached], label="Semi-Detached", marker='o')
    plt.plot(range(12), [total/terraced_count for total in total_terraced], label="Terraced", marker='o')
    
    plt.title('Average Load per day')
    plt.xlabel('Month of the Year')
    plt.ylabel('Daily Load (kWh)')
    plt.xticks(range(12), labels=[f'{month}' for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./policy_data/faraday_processed/average_per_month.png')
    plt.show()

def get_load_files():
    """
    Returns the paths to all of the processed load traces.
    """
    files = []
    for type in ["Detached", "Semi-detached", "Terraced"]:
        files = files + [f"./policy_data/faraday_processed/{type}/{f}" for f in os.listdir(f"./policy_data/faraday_processed/{type}") if f.endswith('.txt')]
    return files

if __name__ == "__main__":
    # Create output directory
    shutil.rmtree("./policy_data/faraday_processed/")
    os.makedirs('./policy_data/faraday_processed/', exist_ok=True)

    # First process the raw data
    for load_file_idx in range(365):
        process_responses(f'./policy_data/faraday_raw/day_{load_file_idx}.json')
        
    # Then analyze and plot the processed data
    files = get_load_files()
    analyze_and_plot_hourly(files)
    analyze_and_plot_monthly(files)
