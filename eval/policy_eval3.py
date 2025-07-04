# to compute E and W+E (where E is with T1 and no c and W+E is with T2 and no c). Note modified operation no_ev.
import subprocess
import pandas as pd
import re
import os

# Function to get paths of house files for each archetype
def get_house_files(archetype):
    # Use only A_House and D_House files for each archetype
    return [f"A_House_{i}.txt" for i in range(1, 60)] + [f"D_House_{i}.txt" for i in range(1, 42)]

# Base path for the files
base_path = os.path.abspath("./policy_data")  # This makes the base path absolute

# Archetypes and their corresponding folders
archetypes = ["Detached", "Semi_Detached", "Terraced"]
operations = ["no_ev"]
wfh_types = ["T1", "T2"]
solar_conditions = {"worst": "Lerwick_pv.txt", "best": "Weymouth_pv.txt"}

# Prepare DataFrame to collect results
results = []

# Iterate over configurations
for archetype in archetypes:
    for house_file in get_house_files(archetype):
        house_file_path = os.path.join(base_path, archetype, house_file)
        
        # Check if the house file exists
        if not os.path.exists(house_file_path):
            print(f"Error: File does not exist - {house_file_path}")
            continue  # Skip this file if it does not exist

        for wfh_type in wfh_types:
            for op in operations:
                for solar_key, solar_file_name in solar_conditions.items():
                    solar_file_path = os.path.join(base_path, "Solar_UK", solar_file_name)
                    
                    # Check if the solar file exists
                    if not os.path.exists(solar_file_path):
                        print(f"Error: Solar file does not exist - {solar_file_path}")
                        continue  # Skip this solar file if it does not exist
                    
                    # Construct command
                    command = f"./bin/sim 2100 480 10 20 1 0.5 0.95 365 {house_file_path} {solar_file_path} 0.8 0.2 60.0 7.4 {op} {base_path}/ev_UK/merged_{wfh_type}_UK.csv 0 4"

                    print("Executing command: " + command)
                    
                    # Execute the command
                    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)

                    # Extract numbers from output
                    grid_import_match = re.search(r"Grid import: (\d+\.?\d*)", result.stdout)
                    total_load_match = re.search(r"Total load: (\d+\.?\d*)", result.stdout)
                    
                    # Store results
                    results.append({
                        "Archetype": archetype,
                        "House number": house_file.split('_')[2].split('.')[0],
                        "WFH Type": wfh_type,
                        "Operation": op,
                        "Solar": solar_key,
                        "Grid Import": float(grid_import_match.group(1)) if grid_import_match else None,
                        "Total Load": float(total_load_match.group(1)) if total_load_match else None
                    })

# Convert results to DataFrame and save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("simulation_results_W+P_P.csv", index=False)
