import pandas as pd
import re
import os

# Function to extract information from the filename
def extract_info(filename):
    pattern = r'soc_(.*?)_(\d+)_(\d+)\.txt'
    match = re.search(pattern, filename)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Operation Policy", "WFH", "Average SOC", "<50%", "<40%", "<30%"])

# Process each file in the current directory
for filename in os.listdir('.'):
    if filename.startswith('soc_') and filename.endswith('.txt'):
        op_policy, load_number, wfh_type = extract_info(filename)
        
        # Read SOC values from the file
        with open(filename, 'r') as file:
            soc_values = [float(line.strip()) for line in file]
        
        # Compute statistics
        average_soc = sum(soc_values) / len(soc_values)
        below_50_percent = sum(soc <= 20 for soc in soc_values) / len(soc_values)
        below_40_percent = sum(soc <= 16 for soc in soc_values) / len(soc_values)
        below_30_percent = sum(soc <= 12 for soc in soc_values) / len(soc_values)
        
        # Append the results to the DataFrame
        results_df = results_df.append({
            "Operation Policy": op_policy,
            "WFH": wfh_type,
            "Average SOC": average_soc,
            "<50%": below_50_percent,
            "<40%": below_40_percent,
            "<30%": below_30_percent
        }, ignore_index=True)

# Group by Operation Policy and WFH, and calculate the average values
avg_results_df = results_df.groupby(["Operation Policy", "WFH"]).mean().reset_index()

# Save the averaged results to a CSV file
avg_results_df.to_csv("soc_statistics_20.csv", index=False)
