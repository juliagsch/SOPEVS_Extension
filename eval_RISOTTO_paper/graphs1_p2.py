import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('evaluation_results.csv')

# Convert numeric columns to float
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

# Filter for specific operation policies
operation_policies = [ 'optimal_unidirectional', 'hybrid_unidirectional', 'safe_unidirectional', 'optimal_bidirectional', 'hybrid_bidirectional', 'safe_bidirectional'  ]
df_filtered = df[df['Operation Policy'].isin(operation_policies)]

# Create a multi-level bar chart using seaborn
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x='Operation Policy', y='Cost', hue='WFH Type', data=df_filtered, ci=None)

# Add cost values above the bars
for p in bar_plot.patches:
    bar_plot.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points')

plt.title('Cost by Operation Policy and WFH Type')
plt.xlabel('Operation Policy')
plt.ylabel('Cost')
plt.legend(title='WFH Type')
plt.tight_layout()
plt.show()
