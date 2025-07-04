import pandas as pd

# Load the CSV data into a DataFrame
data = pd.read_csv('emissions_all_scenarios.csv')

# Define the number of households for each archetype
household_numbers = {
    'Terraced': 6417000,
    'S-Detached': 5810000,
    'Detached': 4137000
}

# Define conversion rates
conversion_rates = list(range(0, 101, 5))  # From 0 to 100 in steps of 5%

# Define the scenarios to include in the output
included_scenarios = ["H+E", "H+P = P", "E+P", "H+E+P", "H+P+E+S"]

# Process the data to compute total emissions for each operation and solar combination
def compute_total_emissions(operation, solar):
    results = []
    for rate in conversion_rates:
        p = rate / 100  # Convert percentage to proportion
        row = {'Conversion Rate (%)': rate}
        for scenario in included_scenarios:
            total_emissions = 0
            for archetype in ['Terraced', 'S-Detached', 'Detached']:
                try:
                    # Extract emissions data for the current scenario
                    emissions_op_solar = data.loc[(data['Archetype'] == archetype) & 
                                                  (data['Operation'] == operation) & 
                                                  (data['Solar'] == solar) & 
                                                  (data['Scenario'] == scenario), 'Emissions (kg CO2)'].values[0]
                except IndexError:
                    print(f"No data for {archetype}, {operation}, {solar}, {scenario}")
                    emissions_op_solar = 0  # Default to zero if no data found

                try:
                    emissions_nc = data.loc[(data['Archetype'] == archetype) & 
                                            (data['Scenario'] == 'W'), 'Emissions (kg CO2)'].values[0]
                    emissions_nc = emissions_nc + 319.5  # Add the emissions for the new FF emissions, as it was 86.5 before ans id 406 now
                except IndexError:
                    print(f"No data for {archetype}, 'not-converted'")
                    emissions_nc = 0  # Default to zero if no data found

                # Calculate emissions for the current archetype
                emissions_archetype = p * household_numbers[archetype] * emissions_op_solar + (1 - p) * household_numbers[archetype] * emissions_nc
                total_emissions += emissions_archetype
            
            # Convert kg to megatons
            total_emissions_megatons = round(total_emissions / 1e6)
            row[f'{scenario}'] = total_emissions_megatons
        
        results.append(row)
    
    # Output to CSV
    df = pd.DataFrame(results)
    filename = f'2507_e_{operation}_{solar}_all_scenarios.csv'
    df.to_csv(filename, index=False)
    print(f'Output file {filename} has been created.')

# Generate output files for each operation and solar scenario
compute_total_emissions('uni', 'best')
compute_total_emissions('uni', 'worst')
compute_total_emissions('bi', 'best')
compute_total_emissions('bi', 'worst')
