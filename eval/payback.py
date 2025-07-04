import pandas as pd

# Load the data from the CSV file that gives OPEX for each archetype x WFH x operation x solar
file_path = 'averaged_simulation_results.csv'
data = pd.read_csv(file_path)

# Define pre-conversion OPEX for each archetype (these values are independent of WFH, operation and solar)
pre_conversion_opex = {
    'Terraced': 2739.301,
    'Semi_Detached': 2904.193,
    'Detached': 2957.12
}


# Initialize a list to hold the results
results = []

# Iterate over each row in the dataframe to compute OPEX savings
for index, row in data.iterrows():
    archetype = row['Archetype']
    wfh_type = row['WFH Type']
    operation = row['Operation']
    solar = row['Solar']
    grid_cost = row['Grid Cost']
    
    # Calculate OPEX Savings
    opex_savings = pre_conversion_opex[archetype] - grid_cost
    
    # Append the results
    results.append({
        'Archetype': archetype,
        'WFH Type': wfh_type,
        'Operation': operation,
        'Solar': solar,
        'OPEX Savings': opex_savings
    })

# Convert the list of results into a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('opex_savings_results.csv', index=False)

print("CSV file with OPEX Savings has been created.")
