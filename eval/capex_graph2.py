#This is the right file to use for the paper, with uni and bi in the correct order.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'terraced_capex_avg.csv'  # Adjust this path if necessary
data = pd.read_csv(file_path)

# Extract 'unidirectional' or 'bidirectional' from the 'Operation' column and ensure it's categorical with a specified order
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Unidirectional' if 'unidirectional' in x.lower() else 'Bidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], categories=['Unidirectional', 'Bidirectional'], ordered=True)

# Calculate the maximum CAPEX for scaling y-axis uniformly across all plots
max_cost = data['Cost'].max()

def create_capex_bar_chart(archetype):
    # Filter data by archetype
    archetype_data = data[data['Archetype'] == archetype]

    # Separate best and worst scenarios for CAPEX
    best_data = archetype_data[archetype_data['Solar'] == 'best'].copy()
    worst_data = archetype_data[archetype_data['Solar'] == 'worst'].copy()

    # Merge on policy and CAH Type
    merged_data = pd.merge(best_data, worst_data, on=['Operation Policy', 'CAH Type'], suffixes=('_best', '_worst'))
    
    # Ensure not to proceed if merged data is empty
    if merged_data.empty:
        print(f"No data available for {archetype} with specified criteria.")
        return
    
    # Custom color palette (old color scheme)
    custom_colors = ['#73C5C5', '#9fc5e8', '#BDE2B9']  # Light blue, Sapphire, Steel blue

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Operation Policy', y='Cost_best', hue='CAH Type', data=merged_data, palette=custom_colors)

    plt.ylabel('CAPEX (£)')
    #plt.title(f'CAPEX for {archetype} Houses')

    # Adding error bars and annotations for best and worst values
    for i, row in merged_data.iterrows():
        best_val = row['Cost_best']
        worst_val = row['Cost_worst']
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
        plt.text(x, height - offset, f'{int(best_val)}', ha="center", va='top', fontsize=10, color='black')
        plt.text(x, height + ci, f'{int(worst_val)}', ha="center", va='bottom', fontsize=10)

    # Set y-axis limit to the previously calculated max plus 20%
    plt.ylim(0, max_cost + 0.2 * max_cost)

    # Adjust the legend position
    plt.legend(title='CAH Type', loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# Call function for each archetype
create_capex_bar_chart('Terraced')
