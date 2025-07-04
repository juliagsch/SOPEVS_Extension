import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_process_hourly(file_path):
    with open(file_path, 'r') as file:
        # Read all lines and convert them to floats
        data = np.array([float(line.strip()) for line in file.readlines()])
        
    # Calculate total PV generation
    total_pv_generation = np.sum(data)
    
    # Calculate average hourly production
    hourly_averages = [np.mean(data[hour::24]) for hour in range(24)]
    
    return total_pv_generation, hourly_averages

def read_and_process_monthly(file_path):
    with open(file_path, 'r') as file:
        # Read all lines and convert them to floats
        data = np.array([float(line.strip()) for line in file.readlines()])
        
    # Calculate total PV generation
    total_pv_generation = np.sum(data)
    
    # Assume data has 365 days, so we split it into 12 months
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    monthly_averages = []
    start_idx = 0
    for days in days_in_month:
        end_idx = start_idx + days
        monthly_averages.append(np.sum(data[start_idx * 24:end_idx * 24])/days)  # Average over the month
        start_idx = end_idx  # Move to the next month
    
    return total_pv_generation, monthly_averages

def analyze_and_plot_daily(solar_files):
    plt.figure(figsize=(10, 5))

    for solar_file in solar_files:
        _, hourly = read_and_process_hourly(solar_file)
        plt.plot(range(24), hourly, label=solar_file.capitalize(), marker='o')
    
    plt.title('Average Hourly PV Production')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Production (kWh)')
    plt.xticks(range(24), labels=[f'{hour}:00' for hour in range(24)], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./policy_data/Solar_UK/average_daily_pv_production.png')
    plt.show()

def analyze_and_plot_monthly(solar_files):
    plt.figure(figsize=(10, 5))

    for solar_file in solar_files:
        _, monthly = read_and_process_monthly(solar_file)
        plt.plot(range(12), monthly, label=solar_file.capitalize(), marker='o')
    
    plt.title('Average Monthly PV Production')
    plt.xlabel('Month of the Year')
    plt.ylabel('Average Production per Day (kWh)')
    plt.xticks(range(12), labels=[f'{month}' for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./policy_data/Solar_UK/average_monthly_pv_production.png')
    plt.show()

def get_solar_files():
    # Changed path to go up one level
    return [f"./policy_data/Solar_UK/{f}" for f in os.listdir("./policy_data/Solar_UK") if f.endswith('.txt')]

if __name__ == "__main__":
    solar_files = get_solar_files()
    # Then analyze and plot the processed data
    analyze_and_plot_daily(solar_files)
    analyze_and_plot_monthly(solar_files)
