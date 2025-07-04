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

# Read the results from the previous execution
results_file = "evaluation_results_20_interm_0.5.csv"
results_df = pd.read_csv(results_file)

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
    # Uncomment the following line to execute the command in your local environment
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    # Process the output as needed
    #print(result.stdout.decode('utf-8'))

# Optionally, save any new results or perform further processing
