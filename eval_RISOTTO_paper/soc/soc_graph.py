import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data into a pandas DataFrame
data = pd.read_csv('soc_statistics_20.csv')

# Define the color palette
palette = "rocket"

# Abbreviated policy names for x-axis labels
policy_labels = {
    "safe_bidirectional": "N-B",
    "safe_unidirectional": "N-U.",
    "hybrid_bidirectional": "SG-B",
    "hybrid_unidirectional": "SG-U",
     "optimal_bidirectional": "SO-B",
    "optimal_unidirectional": "SO-U"
   
}

# Set the font scale and style
sns.set(font_scale=0.8)
sns.set_style("white")  # Set style to white to remove background

# Create individual plots for each WFH type
for wfh_type in [1, 23, 3]:
    # Filter data for the current WFH type
    wfh_data = data[data['WFH'] == wfh_type]
    
    # Create a new figure for the current WFH type
    plt.figure(figsize=(8, 4))
    
    # Create a bar plot
    sns.barplot(x=wfh_data['Operation Policy'].map(policy_labels), y=wfh_data['Average SOC'], palette=palette)
    plt.xlabel("Operation Policy", fontsize=16)
    plt.ylabel("Average SOC", fontsize=16)

    # Display average SOC values above the bars with a smaller font size
    for j, v in enumerate(wfh_data['Average SOC']):
        plt.text(j, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=16)

    # Adjust the y-axis limit based on your data range
    plt.xticks(rotation=0, fontsize=16)  # Ensure horizontal x-axis labels
    plt.ylim(0, data['Average SOC'].max() + 6)  # +2 for a little headroom above the highest bar
    plt.tight_layout()
    plt.show()
