import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'averaged_simulation_results.csv'  # Update this path as necessary
data = pd.read_csv(file_path)

# Adjust 'Operation' to policy categories
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Unidirectional' if 'unidirectional' in x.lower() else 'Bidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], ['Unidirectional', 'Bidirectional'], ordered=True)  # Ensure ordering

# Convert Grid Emissions to kilograms and round
# !! need to make sure that the input emissions are really in gCo2
data['Grid Emissions kg'] = (data['Grid Emissions'] / 1000).round().astype(int)

# Find the maximum y-value to set a uniform y-axis
max_emissions = data['Grid Emissions kg'].max()

def create_confidence_bar_chart(archetype):
    # Filter data by archetype
    archetype_data = data[data['Archetype'] == archetype]

    # Separate best and worst scenarios
    best_data = archetype_data[archetype_data['Solar'] == 'best'].copy()
    worst_data = archetype_data[archetype_data['Solar'] == 'worst'].copy()

    # Merge on policy and CAH Type
    merged_data = pd.merge(best_data, worst_data, on=['Operation Policy', 'CAH Type'], suffixes=('_best', '_worst'))

    # Ensure not to proceed if merged data is empty
    if merged_data.empty:
        print(f"No data available for {archetype} with specified criteria.")
        return

    # Custom color palette (old color scheme)
    custom_colors = ['#80BCBD', '#AAD9BB', '#D5F0C1']

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Operation Policy', y='Grid Emissions kg_best', hue='CAH Type', data=merged_data, palette=custom_colors)

    plt.ylabel('CO2 Emissions (kg)')
    #plt.title(f'CO2 Emissions for {archetype} Houses')

    # Adding error bars and annotations for best and worst values
    for i, row in merged_data.iterrows():
        best_val = row['Grid Emissions kg_best']
        worst_val = row['Grid Emissions kg_worst']
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

    # Set uniform y-axis across all charts
    plt.ylim(0, max_emissions + 0.2 * max_emissions)  # Increase y-axis limit by 20%

    # Adjust the legend position
    plt.legend(title='CAH Type', loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Call function for each archetype
create_confidence_bar_chart('Detached')
create_confidence_bar_chart('Semi_Detached')
create_confidence_bar_chart('Terraced')
