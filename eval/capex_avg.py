import pandas as pd

# Read the input CSV file
input_file = 's_detached_capex_single.csv'
df = pd.read_csv(input_file)

# Group by 'WFH Type', 'Operation', and 'Solar' and calculate the mean of 'Battery', 'PV', and 'Cost'
grouped_df = df.groupby(['WFH Type', 'Operation', 'Solar']).mean().reset_index()

# Select only the required columns
output_df = grouped_df[['WFH Type', 'Operation', 'Solar', 'Battery', 'PV', 'Cost']]

# Write the output to a new CSV file
output_file = 's_detached_capex_avg.csv'
output_df.to_csv(output_file, index=False)

print(f"Output file '{output_file}' has been created.")
