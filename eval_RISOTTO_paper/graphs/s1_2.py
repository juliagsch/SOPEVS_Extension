import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# epsilon 0.5
# Load the data from the CSV file
file_path = 'averaged_evaluation_results_10.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Filter only unidirectional policies
#data = data[data['Operation Policy'].str.contains("unidirectional")]

# Create a seaborn multilevel bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Operation Policy', y='Cost', hue='WFH Type', data=data)

# Additional customization
plt.title('Average Cost by Operation Policy and WFH Type')
plt.ylabel('Average Cost')
plt.xlabel('Operation Policy')
plt.xticks(rotation=45)
plt.legend(title='WFH Type')

# Display the plot
plt.tight_layout()
plt.show()
