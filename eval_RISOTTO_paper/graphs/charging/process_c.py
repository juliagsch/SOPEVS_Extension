import pandas as pd
import re
import os

# Function to extract information from the filename
def extract_info(filename):
    pattern = r'c_(.*?)_(\d+)_(\d+)\.txt'
    match = re.search(pattern, filename)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Operation Policy", "WFH", "Hour", "Average Charging Amount"])
print(results_df)

# Process each file in the current directory
for filename in os.listdir('.'):
    if filename.startswith('c_') and filename.endswith('.txt'):
        op_policy, load_number, wfh_type = extract_info(filename)
        
        # Read charging values from the file
        charging_data = pd.read_csv(filename, delim_whitespace=True, names=["Hour", "Charging Amount"])
        print(filename)
        print(charging_data)
        
        # Group by hour and compute the average charging amount
        avg_charging = charging_data.groupby("Hour").mean().reset_index()
        avg_charging["Operation Policy"] = op_policy
        avg_charging["WFH"] = wfh_type

        # Append the results to the DataFrame
        results_df = pd.concat([results_df, avg_charging])

# Group by Operation Policy, WFH, and Hour, and calculate the average values
avg_results_df = results_df.groupby(["Operation Policy", "WFH", "Hour"]).mean().reset_index()

# Save the averaged results to a CSV file
avg_results_df.to_csv("charging_statistics_20.csv", index=False)
