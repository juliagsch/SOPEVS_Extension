import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'averaged_simulation_results.csv'  # Adjust this path if necessary
data = pd.read_csv(file_path)

# Extract 'unidirectional' or 'bidirectional' from the 'Operation' column and ensure it's categorical with a specified order
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Bidirectional' if 'bidirectional' in x.lower() else 'Unidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], categories=['Unidirectional', 'Bidirectional'], ordered=True)

# Convert Grid Cost to pounds and round to the nearest integer
data['OPEX (£)'] = data['Grid Cost'].round().astype(int)

# Calculate the maximum OPEX for scaling y-axis uniformly across all plots
max_opex = data.groupby(['Operation Policy', 'WFH Type']).agg({'OPEX (£)': 'max'}).max().values[0]

def create_opex_bar_chart(archetype):
    # Filter data by archetype
    archetype_data = data[data['Archetype'] == archetype]

    # Separate best and worst scenarios for OPEX
    best_data = archetype_data[archetype_data['Solar'] == 'best'].copy()
    worst_data = archetype_data[archetype_data['Solar'] == 'worst'].copy()

    # Merge on policy and WFH Type
    merged_data = pd.merge(best_data, worst_data, on=['Operation Policy', 'WFH Type'], suffixes=('_best', '_worst'))
    
    # Ensure not to proceed if merged data is empty
    if merged_data.empty:
        print(f"No data available for {archetype} with specified criteria.")
        return
    
    # Custom color palette
    custom_colors = ['#C4DDFF', '#7FB5FF', '#79DAE8']

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Operation Policy', y='OPEX (£)_best', hue='WFH Type', data=merged_data, palette=custom_colors)

    plt.ylabel('OPEX (£)')

    # Adding error bars and annotations for best and worst values
    for p, (best_val, worst_val) in zip(bar_plot.patches, zip(merged_data['OPEX (£)_best'], merged_data['OPEX (£)_worst'])):
        height = p.get_height()
        ci = worst_val - best_val
        plt.errorbar(p.get_x() + p.get_width() / 2., height, yerr=[[0], [ci]], fmt='none', capsize=5, color='black', lw=2)
        offset = max(0.03 * height, 10)
        best_val_position = height - offset
        plt.text(p.get_x() + p.get_width() / 2., best_val_position, f'{best_val}', ha="center", va='top', fontsize=10, color='black')
        plt.text(p.get_x() + p.get_width() / 2., height + ci, f'{worst_val}', ha="center", va='bottom', fontsize=10)

    # Set y-axis limit to the previously calculated max plus 20%
    plt.ylim(0, max_opex + 0.2 * max_opex)

    # Adjust the legend position
    plt.legend(title='WFH Type', loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Call function for each archetype
create_opex_bar_chart('Detached')
create_opex_bar_chart('Semi_Detached')
create_opex_bar_chart('Terraced')
