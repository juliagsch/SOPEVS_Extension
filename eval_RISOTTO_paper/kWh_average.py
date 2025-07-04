import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('RESULTS.csv')

# Group by 'Operation Policy' and 'WFH Type' and compute the mean for specified columns
aggregated_df = df.groupby(['Operation Policy', 'WFH Type']).agg({
    'ev_charged': 'mean',
    'ev_discharged': 'mean',
    'stat_charged': 'mean',
    'stat_discharged': 'mean'
}).reset_index()

# Adjust the relevant columns by dividing by 100 and rounding to 1 decimal point
for column in ['ev_charged', 'ev_discharged', 'stat_charged', 'stat_discharged']:
    aggregated_df[column] = aggregated_df[column].apply(lambda x: round(x / 100, 1))
   # aggregated_df[column] = aggregated_df[column].apply(lambda x: round(x , 1))

# Save the aggregated DataFrame to a new CSV file
aggregated_df.to_csv('RESULTS_2.csv', index=False)

print("The adjusted aggregation is complete and saved to RESULTS_2.csv")
