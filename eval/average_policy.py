# Need to modfiy this if there is an EV fpr the grid emissions caomputation (is FF added or not?)
import pandas as pd
# uncomment l.15 if there is no EV to add FF car emissions, comment l14
# Load the simulation results CSV
df = pd.read_csv("simulation_results_2607__opex2.csv")

# Define the conversion rates
pounds_per_kwh = 0.35
gCO2_per_kwh = 162

# Calculate additional columns
df['Grid Cost'] = df['Grid Import'] * pounds_per_kwh
# Modify this is there is an EV
df['Grid Emissions'] = df['Grid Import'] * gCO2_per_kwh / 1000 
#df['Grid Emissions'] = df['Grid Import'] * gCO2_per_kwh / 1000 + 406
#df['Grid Emissions'] = df['Grid Import'] * gCO2_per_kwh + 86.5
#df['Indepence'] = (100 - (df['Grid Import'] / df['Total Load']) * 100)

# Group by the necessary columns and calculate the mean for each group
results = df.groupby(['Archetype', 'WFH Type', 'Operation', 'Solar']).mean().reset_index()

# Round all values to the nearest integer
results = results.round(0)

# Cast to int to remove any trailing .0 after rounding
results['Grid Import'] = results['Grid Import'].astype(int)
results['Total Cost'] = results['Total Cost'].astype(int)
results['Grid Cost'] = results['Grid Cost'].astype(int)
results['Grid Emissions'] = results['Grid Emissions'].astype(int)
#results['Indepence'] = results['Indepence'].astype(int)

# Select only the required columns to match the output file specification
#results = results[['Archetype', 'WFH Type', 'Operation', 'Solar', 'Grid Import', 'Total Load', 'Grid Cost', 'Grid Emissions', 'Indepence']]
results = results[['Archetype', 'WFH Type', 'Operation', 'Solar', 'Grid Import', 'Total Cost', 'Grid Cost', 'Grid Emissions']]


# Save the averaged results to a new CSV file
results.to_csv('2607_averaged_simulation_results_2607__opex2.csv', index=False)
