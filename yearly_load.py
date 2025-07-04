import os
import numpy as np

# List of directories to analyze
directories = ['policy_data/Terraced', 'policy_data/Semi_Detached', 'policy_data/Detached']

# Emissions factor for electricity in gCO2 per kWh
gCO2_per_kwh = 162

# Emissions from a single personal car per year in kgCO2e
car_emissions_per_year = 86.5

def process_directory(directory):
    total_loads = []

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return 0, 0  # Return 0s if the directory does not exist

    # Traverse through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as f:
                    loads = np.array([float(line.strip()) for line in f.readlines()])
                    total_loads.append(loads)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    # Aggregate the data
    if total_loads:
        total_load = np.sum(total_loads, axis=0)
        annual_average = np.sum(total_load) / len(total_loads)  # Average over the number of files
        # Calculate total yearly emissions from electricity use in kg
        yearly_emissions = (annual_average * gCO2_per_kwh) / 1000  # Convert from g to kg
        # Add car emissions to the load emissions
        total_emissions_with_car = yearly_emissions + car_emissions_per_year
        return yearly_emissions, total_emissions_with_car
    else:
        print(f"No load data found in {directory}.")
        return 0, 0  # Return 0 if no loads were found

# Process each directory and store results
results = {}
for directory in directories:
    yearly_emissions, total_emissions_with_car = process_directory(directory)
    results[directory] = {
        'Yearly Load Emissions (kg CO2)': yearly_emissions,
        'Total Emissions with Car (kg CO2)': total_emissions_with_car
    }

# Output the results
for directory, emissions in results.items():
    category = directory.split('/')[-1]  # Get the last part of the directory path
    print(f"{category} Houses - Yearly Load Emissions: {emissions['Yearly Load Emissions (kg CO2)']:.2f} kg CO2")
    print(f"{category} Houses - Total Emissions with Car: {emissions['Total Emissions with Car (kg CO2)']:.2f} kg CO2")
