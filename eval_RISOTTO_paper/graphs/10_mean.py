import pandas as pd

# Sample data (replace this with your actual data)
data = """
Battery: , PV: , Cost: , WFH Type: T1, Operation Policy: optimal_unidirectional


"""

# Split the data into lines
lines = data.strip().split('\n')

# Parse each line into a dictionary and append to a list
records = []
for line in lines:
    parts = line.split(',')
    record = {
        'Battery': float(parts[0].split(':')[1].strip()),
        'PV': float(parts[1].split(':')[1].strip()),
        'Cost': float(parts[2].split(':')[1].strip()),
        'WFH Type': parts[3].split(':')[1].strip(),
        'Operation Policy': parts[4].split(':')[1].strip()
    }
    records.append(record)

# Create a DataFrame from the records
df = pd.DataFrame(records)

# Group by WFH Type and Operation Policy, then calculate averages
grouped_df = df.groupby(['WFH Type', 'Operation Policy']).mean().reset_index()

# Save the results to a CSV file
grouped_df.to_csv('grouped_evaluation_results.csv', index=False)
