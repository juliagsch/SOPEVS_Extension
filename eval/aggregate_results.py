import pandas as pd

# Load the CSV data into a DataFrame
data = pd.read_csv('emissions.csv')

# Define the number of households for each archetype
household_numbers = {
    'Terraced': 6417000,
    'Semi-Detached': 5810000,
    'Detached': 4137000
}

# Define conversion rates
conversion_rates = list(range(0, 101, 5))  # From 0 to 100 in steps of 5%

# Process the data to compute total emissions for each scenario
def compute_total_emissions(operation, solar):
    results = []
    for rate in conversion_rates:
        p = rate / 100  # Convert percentage to proportion
        total_emissions = 0
        for archetype in ['Terraced', 'Semi-Detached', 'Detached']:
            # Extract emissions data for the current scenario
            emissions_op_solar = data.loc[(data['Archetype'] == archetype) & 
                                          (data['Operation'] == operation) & 
                                          (data['Solar'] == solar), 'Emissions (kgCO2)'].values[0]
            emissions_nc = data.loc[(data['Archetype'] == archetype) & 
                                    (data['Solar'] == 'not-converted'), 'Emissions (kgCO2)'].values[0]
            #print(emissions_op_solar, emissions_nc)

            # Calculate emissions for the current archetype
            emissions_archetype = p * household_numbers[archetype] * emissions_op_solar + (1 - p) * household_numbers[archetype] * emissions_nc
            total_emissions += emissions_archetype
        
        # Convert kg to megatons
        total_emissions_gigatons = round(total_emissions / 1000)
        results.append({'Conversion Rate (%)': rate, 'Total Emissions T': total_emissions_gigatons})
    
    # Output to CSV
    df = pd.DataFrame(results)
    filename = f'e_{operation}_{solar}.csv'
    df.to_csv(filename, index=False)
    print(f'Output file {filename} has been created.')

# Generate output files for each scenario
compute_total_emissions('uni', 'best')
compute_total_emissions('uni', 'worst')
compute_total_emissions('bi', 'best')
compute_total_emissions('bi', 'worst')
