# Correct one
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file, normal tariff
file_path = 'averaged_simulation_results.csv'  # Adjust this path if necessary
# below is the file path to data with LBN tariff 
#file_path_LBN = '2607_averaged_simulation_results_2607__opex2.csv'

data = pd.read_csv(file_path)

# Extract 'unidirectional' or 'bidirectional' from the 'Operation' column and ensure it's categorical with a specified order
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Unidirectional' if 'unidirectional' in x.lower() else 'Bidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], categories=['Unidirectional', 'Bidirectional'], ordered=True)

# Convert Grid Cost to pounds and round to the nearest integer
data['OPEX (£)'] = data['Grid Cost'].round().astype(int)

# Calculate the maximum OPEX for scaling y-axis uniformly across all plots
max_opex = data['OPEX (£)'].max()

def create_opex_bar_chart(archetype):
    # Filter data by archetype
    archetype_data = data[data['Archetype'] == archetype]

    # Separate best and worst scenarios for OPEX
    best_data = archetype_data[archetype_data['Solar'] == 'best'].copy()
    worst_data = archetype_data[archetype_data['Solar'] == 'worst'].copy()

    # Merge on policy and CAH Type
    merged_data = pd.merge(best_data, worst_data, on=['Operation Policy', 'CAH Type'], suffixes=('_best', '_worst'))
    
    # Ensure not to proceed if merged data is empty
    if merged_data.empty:
        print(f"No data available for {archetype} with specified criteria.")
        return
    
    # Custom color palette
    custom_colors = ['#C4DDFF', '#7FB5FF', '#79DAE8']

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Operation Policy', y='OPEX (£)_best', hue='CAH Type', data=merged_data, palette=custom_colors)

    plt.ylabel('OPEX (£)')

    # Adding error bars and annotations for best and worst values
    for i, row in merged_data.iterrows():
        best_val = row['OPEX (£)_best']
        worst_val = row['OPEX (£)_worst']
        height = best_val
        ci = worst_val - best_val

        # Find the corresponding patch
        patches = [patch for patch in bar_plot.patches if round(patch.get_height(), 2) == round(height, 2)]
        if not patches:
            continue
        patch = patches[0]
        x = patch.get_x() + patch.get_width() / 2

        plt.errorbar(x, height, yerr=[[0], [ci]], fmt='none', capsize=5, color='black', lw=2)
        offset = max(0.03 * height, 10)
        plt.text(x, height - offset, f'{best_val}', ha="center", va='top', fontsize=10, color='black')
        plt.text(x, height + ci, f'{worst_val}', ha="center", va='bottom', fontsize=10)

    # Set y-axis limit to the previously calculated max plus 20%
    plt.ylim(0, max_opex + 0.2 * max_opex)

    # Adjust the legend position
    plt.legend(title='CAH Type', loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Call function for each archetype
create_opex_bar_chart('Detached')
create_opex_bar_chart('Semi_Detached')
create_opex_bar_chart('Terraced')

