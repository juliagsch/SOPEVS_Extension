import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data into a pandas DataFrame
data = pd.read_csv('charging_statistics.csv')

# Define the color palette
palette = "rocket"

# Filter data for WFH T1
data_t1 = data[data['WFH'] == 23]

# Filter data for unidirectional policies
unidirectional_policies = ['hybrid_unidirectional', 'optimal_unidirectional', 'safe_unidirectional']
data_t1_unidirectional = data_t1[data_t1['Operation Policy'].isin(unidirectional_policies)]

# Filter data for bidirectional policies
bidirectional_policies = ['hybrid_bidirectional', 'optimal_bidirectional', 'safe_bidirectional']
data_t1_bidirectional = data_t1[data_t1['Operation Policy'].isin(bidirectional_policies)]

# Create line plot for Unidirectional policies (WFH T1)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Charging Amount', hue='Operation Policy', data=data_t1_unidirectional, palette=palette)
plt.title('WFH T2 Unidirectional Charging Profile')
plt.xlabel('Hour')
plt.ylabel('Charging Amount (kWh)')
plt.legend(title='Operation Policy', loc='upper left')
plt.show()

# Create line plot for Bidirectional policies (WFH T1)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Charging Amount', hue='Operation Policy', data=data_t1_bidirectional, palette=palette)
plt.title('WFH T2 Bidirectional Charging Profile')
plt.xlabel('Hour')
plt.ylabel('Charging Amount (kWh)')
plt.legend(title='Operation Policy', loc='upper left')
plt.show()
