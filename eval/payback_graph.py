import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
try:
    opex_savings_data = pd.read_csv('opex_savings_results.csv')
    capex_data = pd.read_csv('capex_results.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

merged_data = pd.merge(opex_savings_data, capex_data, on=["Archetype", "CAH Type", "Operation", "Solar"])

merged_data['Payback Time (years)'] = merged_data['Cost'] / merged_data['OPEX Savings']

merged_data['Operation Policy'] = merged_data['Operation'].apply(lambda x: 'Bidirectional' if 'bidirectional' in x else 'Unidirectional')
merged_data['Operation Policy'] = pd.Categorical(merged_data['Operation Policy'], categories=['Unidirectional', 'Bidirectional'], ordered=True)


def create_payback_time_bar_chart(archetype):
    # Filter data by archetype
    archetype_data = merged_data[merged_data['Archetype'] == archetype]
    print(f"Data for {archetype}:")
    print(archetype_data.head())

    # Separate best and worst scenarios for Payback Time
    best_data = archetype_data[archetype_data['Solar'] == 'best'].copy()
    worst_data = archetype_data[archetype_data['Solar'] == 'worst'].copy()

    # Merge on policy and CAH Type
    merged_archetype_data = pd.merge(best_data, worst_data, on=['Operation Policy', 'CAH Type'], suffixes=('_best', '_worst'))
    
    # Ensure not to proceed if merged data is empty
    if merged_archetype_data.empty:
        print(f"No data available for {archetype} with specified criteria.")
        return
    
    # Custom color palette (similar to CAPEX graph)
    custom_colors = ['#73C5C5', '#9fc5e8', '#BDE2B9']  # Light blue, Sapphire, Steel blue

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Operation Policy', y='Payback Time (years)_best', hue='CAH Type', data=merged_archetype_data, palette=custom_colors)

    plt.ylabel('Payback Time (years)')
   # plt.title(f'Payback Time for {archetype} Houses')

    # Adding error bars and annotations for best and worst values
    for i, row in merged_archetype_data.iterrows():
        best_val = row['Payback Time (years)_best']
        worst_val = row['Payback Time (years)_worst']
        height = best_val
        ci = worst_val - best_val
        
        # Find the corresponding patch
        patches = [patch for patch in bar_plot.patches if round(patch.get_height(), 2) == round(height, 2)]
        if not patches:
            continue
        patch = patches[0]
        x = patch.get_x() + patch.get_width() / 2
        
        plt.errorbar(x, height, yerr=[[0], [ci]], fmt='none', capsize=5, color='black', lw=2)
        offset = max(0.03 * height, 0.1)
        plt.text(x, height - offset, f'{best_val:.1f}', ha="center", va='top', fontsize=10, color='black')
        plt.text(x, height + ci + offset, f'{worst_val:.1f}', ha="center", va='bottom', fontsize=10)

    # Set y-axis limit to the previously calculated max plus 20%
    plt.ylim(0, 17)

    # Adjust the legend position
    plt.legend(title='CAH Type', loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Call function for each archetype
archetypes = ['Terraced', 'Detached', 'Semi-Detached']
for archetype in archetypes:
    create_payback_time_bar_chart(archetype)
