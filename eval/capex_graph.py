import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'terraced_capex_avg.csv'  # Adjust this path if necessary
data = pd.read_csv(file_path)

# Extract 'unidirectional' or 'bidirectional' from the 'Operation' column and ensure it's categorical with a specified order
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Unidirectional' if 'unidirectional' in x.lower() else 'Bidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], categories=['Bidirectional', 'Unidirectional'], ordered=True)

# Calculate the maximum CAPEX for scaling y-axis uniformly across all plots
max_cost = data['Cost'].max()

def create_capex_bar_chart(archetype):
    # Filter data by archetype
    archetype_data = data[data['Archetype'] == archetype]

    # Separate best and worst scenarios for CAPEX
    best_data = archetype_data[archetype_data['Solar'] == 'best'].copy()
    worst_data = archetype_data[archetype_data['Solar'] == 'worst'].copy()

    # Merge on policy and WFH Type
    merged_data = pd.merge(best_data, worst_data, on=['Operation Policy', 'WFH Type'], suffixes=('_best', '_worst'))
    
    # Ensure not to proceed if merged data is empty
    if merged_data.empty:
        print(f"No data available for {archetype} with specified criteria.")
        return
    
    # Custom color palette (old color scheme)
    custom_colors = ['#89CFF0', '#0F52BA', '#4682B4']  # Light blue, Sapphire, Steel blue

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Operation Policy', y='Cost_best', hue='WFH Type', data=merged_data, palette=custom_colors)

    plt.ylabel('CAPEX (£)')
    plt.title(f'CAPEX for {archetype} Houses')

    # Adding error bars and annotations for best and worst values
    for p, (best_val, worst_val) in zip(bar_plot.patches, zip(merged_data['Cost_best'], merged_data['Cost_worst'])):
        height = p.get_height()
        ci = worst_val - best_val
        plt.errorbar(p.get_x() + p.get_width() / 2., height, yerr=[[0], [ci]], fmt='none', capsize=5, color='black', lw=2)
        offset = max(0.03 * height, 10)
        best_val_position = height - offset
        plt.text(p.get_x() + p.get_width() / 2., best_val_position, f'{int(best_val)}', ha="center", va='top', fontsize=10, color='black')
        plt.text(p.get_x() + p.get_width() / 2., height + ci, f'{int(worst_val)}', ha="center", va='bottom', fontsize=10)

    # Set y-axis limit to the previously calculated max plus 20%
    plt.ylim(0, max_cost + 0.2 * max_cost)

    # Adjust the legend position
    plt.legend(title='WFH Type', loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Call function for each archetype
create_capex_bar_chart('Terraced')
