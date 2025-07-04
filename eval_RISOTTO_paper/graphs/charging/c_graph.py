import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data into a pandas DataFrame
data = pd.read_csv('charging_statistics_20.csv')

# Define the color palette
palette = "rocket"

# Define the new operation policy labels
policy_labels = {
    "safe_bidirectional": "N-B",
    "safe_unidirectional": "N-U",
    "hybrid_bidirectional": "SG-B",
    "hybrid_unidirectional": "SG-U",
    "optimal_bidirectional": "SO-B",
    "optimal_unidirectional": "SO-U"
}

# Update the 'Operation Policy' column to use the new labels
data['Operation Policy'] = data['Operation Policy'].map(policy_labels)

# Define unidirectional and bidirectional policies
unidirectional_policies = ['N-U', 'SG-U', 'SO-U']
bidirectional_policies = ['N-B', 'SG-B', 'SO-B']

# Filter data for each WFH type and each policy type
for wfh_type in [1, 23, 3]:
    for policy_type, policy_list in [('Unidirectional', unidirectional_policies), ('Bidirectional', bidirectional_policies)]:
        filtered_data = data[(data['WFH'] == wfh_type) & (data['Operation Policy'].isin(policy_list))]
        
        # Create line plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Hour', y='Charging Amount', hue='Operation Policy', data=filtered_data, palette=palette)
        #plt.title(f'WFH T{wfh_type} {policy_type} Charging Profile')
        plt.xlabel('Hour')
        plt.xticks(rotation=0, fontsize=16)
        plt.ylabel('Charging Amount (kWh)')
        plt.show()
