import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
file_path = 'averaged_simulation_results.csv'  # Update this path as necessary
data = pd.read_csv(file_path)

# Adjust 'Operation' to policy categories
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Unidirectional' if 'unidirectional' in x.lower() else 'Bidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], ['Unidirectional', 'Bidirectional'], ordered=True)  # Ensure ordering

# Convert Grid Emissions to kilograms and round
data['Grid Emissions kg'] = (data['Grid Emissions'] / 1000).round().astype(int)

# Find the maximum y-value to set a uniform y-axis
max_emissions = data['Grid Emissions kg'].max()

# Define archetypes and CAH types
archetype_order = ['Detached', 'Semi_Detached', 'Terraced']  # D, SD, T
cah_types = ['H1', 'H2', 'H3']
color_map = {'Unidirectional': '#D5F0C1', 'Bidirectional': '#80BCBD'}

# Function to create a combined CO2 emissions chart
def create_combined_co2_chart():
    plt.figure(figsize=(14, 8))  # Set figure size

    # Initialize positions
    uni_positions = []
    bi_positions = []

    # Loop through each CAH type
    for cah in cah_types:
        for archetype in archetype_order:
            # First loop for Unidirectional
            filtered_uni = data[(data['CAH Type'] == cah) & (data['Operation Policy'] == 'Unidirectional') & (data['Archetype'] == archetype)]

            if len(filtered_uni) >= 2:  # Ensure there are best and worst cases
                best_val_uni = filtered_uni.iloc[0]['Grid Emissions kg']  # Best case
                worst_val_uni = filtered_uni.iloc[1]['Grid Emissions kg']  # Worst case
                ci_uni = worst_val_uni - best_val_uni

                uni_position = len(uni_positions)  # Current position
                uni_positions.append(uni_position)

                # Draw rectangle for Unidirectional
                plt.bar(uni_position, height=ci_uni, bottom=best_val_uni, width=0.4, color=color_map['Unidirectional'], alpha=0.7)

            # Second loop for Bidirectional
            filtered_bi = data[(data['CAH Type'] == cah) & (data['Operation Policy'] == 'Bidirectional') & (data['Archetype'] == archetype)]

            if len(filtered_bi) >= 2:  # Ensure there are best and worst cases
                best_val_bi = filtered_bi.iloc[0]['Grid Emissions kg']  # Best case
                worst_val_bi = filtered_bi.iloc[1]['Grid Emissions kg']  # Worst case
                ci_bi = worst_val_bi - best_val_bi

                # Align Bidirectional bars directly under Unidirectional bars
                bi_position = uni_position  # Use the same position as Unidirectional
                plt.bar(bi_position, height=ci_bi, bottom=best_val_bi, width=0.4, color=color_map['Bidirectional'], alpha=0.7)

    # Set x-ticks with unique labels for D, SD, T only
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=['D', 'SD', 'T'] * len(cah_types), rotation=0, fontsize=14)

    plt.ylabel('CO2 Emissions (kg)', fontsize=16)  # Increase y-axis label font size
    plt.xlabel('Archetype', fontsize=16)  # Add x-axis label

    # Set y-axis limit for consistency
    plt.ylim(0, max_emissions + 0.2 * max_emissions)  # Increase y-axis limit by 20%

    # Center CAH labels between vertical lines
    plt.text(1.0, max_emissions * 1.15, 'CAH1', ha='center', va='top', fontsize=14, color='black')
    plt.text(4.0, max_emissions * 1.15, 'CAH2', ha='center', va='top', fontsize=14, color='black')
    plt.text(7.0, max_emissions * 1.15, 'CAH3', ha='center', va='top', fontsize=14, color='black')

    # Add vertical lines between CAH types to separate H1, H2, and H3
    plt.axvline(x=2.5, color='black', linestyle='--')  # Centered between T of H1 and D of H2
    plt.axvline(x=5.5, color='black', linestyle='--')  # Centered between T of H2 and D of H3

    # Add legend for EV charging
    legend_labels = [plt.Line2D([0], [0], color=color_map['Unidirectional'], lw=4), 
                     plt.Line2D([0], [0], color=color_map['Bidirectional'], lw=4)]
    plt.legend(legend_labels, ['Unidirectional', 'Bidirectional'], loc='lower left', fontsize=14, title='EV Charging', title_fontsize='15')

    # Increase the font size of y-ticks
    plt.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.show()

# Call the function to create the combined chart
create_combined_co2_chart()
