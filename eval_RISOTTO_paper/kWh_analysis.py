import subprocess
import pandas as pd
import re

# Function to convert WFH type number to corresponding file name part
def wfh_type_to_file_part(wfh_type_number):
    if wfh_type_number == '1':
        return 'T1'
    elif wfh_type_number == '2':
        return 'T23'
    elif wfh_type_number == '3':
        return 'T3'
    return None

# Parse the output to extract ev_charged, ev_discharged, stat_charged, and stat_discharged
def parse_subprocess_output(output):
    # Split the output by tabs, assuming the format is as given in the print statement
    parts = output.strip().split("\t")
    if len(parts) == 4:
        return {
            'ev_charged': float(parts[0]),
            'ev_discharged': float(parts[1]),
            'stat_charged': float(parts[2]),
            'stat_discharged': float(parts[3]),
        }
    else:
        return {
            'ev_charged': None,
            'ev_discharged': None,
            'stat_charged': None,
            'stat_discharged': None,
        }

# Read the results from the previous execution
results_file = "evaluation_results_20_interm_0.5.csv"
results_df = pd.read_csv(results_file)

# Prepare a DataFrame for the new results
new_results_df = pd.DataFrame(columns=['Number', 'Operation Policy', 'WFH Type', 'Battery', 'PV', 'ev_charged', 'ev_discharged', 'stat_charged', 'stat_discharged'])

# Base path for the files
base_path = "pecan/"

# Define the command template
command_template = "./bin/sim 1250 460 10 20 1 0.5 0.95 100 {load_file} {pv_file} 0.8 0.2 40.0 7.4 {op} ev_data/{file_name} {battery} {pv}"

# Iterate over each row in the results DataFrame
for index, row in results_df.iterrows():
    load_number = row['Number']
    op = row['Operation Policy']
    wfh_type = wfh_type_to_file_part(str(row['WFH Type']))  # Convert WFH type number to file part
    battery = row['Battery']
    pv = row['PV']

    # Construct load and PV file names
    load_file = f"{base_path}load_{load_number}.txt"
    pv_file = f"{base_path}PV_{load_number}.txt"

    # Construct EV data file name based on WFH type
    ev_file_name = f'ev_merged_{wfh_type}.csv'

    # Construct and execute the command
    command = command_template.format(load_file=load_file, pv_file=pv_file, op=op, file_name=ev_file_name, battery=battery, pv=pv)
    print("Executing command: " + command)
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    
    # Parse the output
    parsed_output = parse_subprocess_output(result.stdout)
    
    # Append the results to the new DataFrame
    new_row = {**row[['Number', 'Operation Policy', 'WFH Type', 'Battery', 'PV']].to_dict(), **parsed_output}
    new_results_df = new_results_df.append(new_row, ignore_index=True)

# Save the new results to a CSV file
new_results_df.to_csv('RESULTS.csv', index=False)
