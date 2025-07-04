import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
#file_path = 'averaged_simulation_results.csv'
#LBN file below
file_path = '2607_averaged_simulation_results_2607__opex2.csv'
data = pd.read_csv(file_path)

# Extract 'Uni' or 'Bi' from the 'Operation' column
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Uni' if 'unidirectional' in x.lower() else 'Bi')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], categories=['Uni', 'Bi'], ordered=True)

# Define archetypes and CAH types
archetype_order = ['Detached', 'Semi_Detached', 'Terraced']  # D, SD, T
cah_types = ['H1', 'H2', 'H3']
color_map = {'Uni': '#C4DDFF', 'Bi': '#79DAE8'}

# Function to create a combined bar chart for all archetypes
def create_combined_opex_chart():
    plt.figure(figsize=(14, 8))  # Set figure size

    # Initialize positions
    uni_positions = []
    bi_positions = []

    # Loop through each CAH type
    for cah in cah_types:
        for archetype in archetype_order:
            # First loop for Uni
            for policy in ['Uni']:
                # Filter data for the current combination
                filtered = data[(data['CAH Type'] == cah) & (data['Operation Policy'] == policy) & (data['Archetype'] == archetype)]

                if len(filtered) >= 2:  # Ensure there are best and worst cases
                    best_val = filtered.iloc[0]['Grid Cost']  # Best case
                    worst_val = filtered.iloc[1]['Grid Cost']  # Worst case
                    ci = worst_val - best_val

                    # Calculate the position for the current bar
                    uni_position = len(uni_positions)  # Current position
                    uni_positions.append(uni_position)

                    # Draw rectangle for Uni
                    plt.bar(uni_position, height=ci, bottom=best_val, width=0.4, color=color_map['Uni'], alpha=0.7)

            # Second loop for Bi, using the same x positions
            for policy in ['Bi']:
                # Filter data for the current combination
                filtered = data[(data['CAH Type'] == cah) & (data['Operation Policy'] == policy) & (data['Archetype'] == archetype)]

                if len(filtered) >= 2:  # Ensure there are best and worst cases
                    best_val = filtered.iloc[0]['Grid Cost']  # Best case
                    worst_val = filtered.iloc[1]['Grid Cost']  # Worst case
                    ci = worst_val - best_val

                    # Use the same position as Uni
                    bi_position = uni_position  # Align Bi bars directly under Uni bars

                    # Draw rectangle for Bi
                    plt.bar(bi_position, height=ci, bottom=best_val, width=0.4, color=color_map['Bi'], alpha=0.7)

    # Set x-ticks with unique labels for D, SD, T only
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=['D', 'SD', 'T'] * len(cah_types), rotation=0, fontsize=14)

    plt.ylabel('OPEX (£)', fontsize=16)  # Increase y-axis label font size
    plt.xlabel('Archetype', fontsize=16)  # Add x-axis label

    # Set y-axis limit for consistency
    plt.ylim(500, 2250)

    # Center CAH labels between vertical lines
    plt.text(1.0, 2100, 'CAH1', ha='center', va='top', fontsize=14, color='black')
    plt.text(4.0, 2100, 'CAH2', ha='center', va='top', fontsize=14, color='black')
    plt.text(7.0, 2100, 'CAH3', ha='center', va='top', fontsize=14, color='black')

    # Add vertical lines between CAH types to separate H1, H2, and H3
    plt.axvline(x=2.5, color='black', linestyle='--')  # Centered between T of H1 and D of H2
    plt.axvline(x=5.5, color='black', linestyle='--')  # Centered between T of H2 and D of H3

    # Add legend for EV charging
    legend_labels = [plt.Line2D([0], [0], color=color_map['Uni'], lw=4), 
                     plt.Line2D([0], [0], color=color_map['Bi'], lw=4)]
    plt.legend(legend_labels, ['Unidirectional', 'Bidirectional'], loc='lower left', fontsize=14, title='EV Charging', title_fontsize='15')

    # Increase the font size of y-ticks
    plt.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.show()

# Call the function to create the combined chart
create_combined_opex_chart()
