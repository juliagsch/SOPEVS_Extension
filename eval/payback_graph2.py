import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
try:
    opex_savings_data = pd.read_csv('opex_savings_results.csv')
    capex_data = pd.read_csv('capex_results.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Merge the data
merged_data = pd.merge(opex_savings_data, capex_data, on=["Archetype", "CAH Type", "Operation", "Solar"])

# Calculate Payback Time
merged_data['Payback Time (years)'] = merged_data['Cost'] / merged_data['OPEX Savings']

# Determine Operation Policy
merged_data['Operation Policy'] = merged_data['Operation'].apply(lambda x: 'Bidirectional' if 'bidirectional' in x else 'Unidirectional')
merged_data['Operation Policy'] = pd.Categorical(merged_data['Operation Policy'], categories=['Unidirectional', 'Bidirectional'], ordered=True)
print(merged_data)

# Define archetypes and CAH types
archetype_order = ['Detached', 'Semi-Detached', 'Terraced']  # D, SD, T
cah_types = ['H1', 'H2', 'H3']
color_map = {'Unidirectional': '#BDE2B9', 'Bidirectional': '#73C5C5'}

# Function to create a combined payback time chart
def create_payback_time_chart():
    plt.figure(figsize=(14, 8))  # Set figure size

    # Initialize positions
    uni_positions = []
    bi_positions = []

    # Loop through each CAH type except H3 first
    for cah in cah_types[:-1]:  # H1 and H2
        for archetype in archetype_order:
            # Initialize the best and worst value holders
            best_val_uni = None
            worst_val_uni = None
            best_val_bi = None
            worst_val_bi = None

            # First loop for Uni
            filtered_uni = merged_data[(merged_data['CAH Type'] == cah) & (merged_data['Operation Policy'] == 'Unidirectional') & (merged_data['Archetype'] == archetype)]
            if len(filtered_uni) >= 2:
                best_val_uni = filtered_uni.iloc[0]['Payback Time (years)']
                worst_val_uni = filtered_uni.iloc[1]['Payback Time (years)']
                ci_uni = worst_val_uni - best_val_uni

                uni_position = len(uni_positions)  # Current position
                uni_positions.append(uni_position)

                # Draw rectangle for Uni
                plt.bar(uni_position, height=ci_uni, bottom=best_val_uni, width=0.4, color=color_map['Unidirectional'], alpha=0.7)

            # Second loop for Bi
            filtered_bi = merged_data[(merged_data['CAH Type'] == cah) & (merged_data['Operation Policy'] == 'Bidirectional') & (merged_data['Archetype'] == archetype)]
            if len(filtered_bi) >= 2:
                best_val_bi = filtered_bi.iloc[0]['Payback Time (years)']
                worst_val_bi = filtered_bi.iloc[1]['Payback Time (years)']
                ci_bi = worst_val_bi - best_val_bi

                # Align Bi bars directly under Uni bars
                bi_position = uni_position  # Use the same position as Uni
                plt.bar(bi_position, height=ci_bi, bottom=best_val_bi, width=0.4, color=color_map['Bidirectional'], alpha=0.7)

    # Now handle CAH H3 separately
    cah = 'H3'
    for archetype in archetype_order:
        # Initialize the best and worst value holders for H3
        best_val_uni = None
        worst_val_uni = None
        best_val_bi = None
        worst_val_bi = None

        # First loop for Uni
        filtered_uni = merged_data[(merged_data['CAH Type'] == cah) & (merged_data['Operation Policy'] == 'Unidirectional') & (merged_data['Archetype'] == archetype)]
        print(filtered_uni)
        if len(filtered_uni) >= 2:
            best_val_uni = filtered_uni.iloc[0]['Payback Time (years)']
            worst_val_uni = filtered_uni.iloc[1]['Payback Time (years)']
            ci_uni = worst_val_uni - best_val_uni

            uni_position = len(uni_positions)  # Current position
            uni_positions.append(uni_position)

            # Draw rectangle for Uni
            plt.bar(uni_position, height=ci_uni, bottom=best_val_uni, width=0.4, color=color_map['Unidirectional'], alpha=0.7)

        # Second loop for Bi
        filtered_bi = merged_data[(merged_data['CAH Type'] == cah) & (merged_data['Operation Policy'] == 'Bidirectional') & (merged_data['Archetype'] == archetype)]
        print(filtered_bi)
        if len(filtered_bi) >= 2:
            best_val_bi = filtered_bi.iloc[0]['Payback Time (years)']
            worst_val_bi = filtered_bi.iloc[1]['Payback Time (years)']
            ci_bi = worst_val_bi - best_val_bi

            # Align Bi bars directly under Uni bars
            bi_position = uni_position  # Use the same position as Uni
            plt.bar(bi_position, height=ci_bi, bottom=best_val_bi, width=0.4, color=color_map['Bidirectional'], alpha=0.7)

    # Set x-ticks with unique labels for D, SD, T only
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=['D', 'SD', 'T'] * len(cah_types), rotation=0)

    plt.ylabel('Payback Time (years)', fontsize=16)  # Increase y-axis label font size
    plt.xlabel('Archetype', fontsize=16)  # Add x-axis label

    # Set y-axis limit for consistency
    plt.ylim(0, 17)

    # Center CAH labels between vertical lines
    plt.text(1.0, 16, 'CAH1', ha='center', va='top', fontsize=14, color='black')
    plt.text(4.0, 16, 'CAH2', ha='center', va='top', fontsize=14, color='black')
    plt.text(7.0, 16, 'CAH3', ha='center', va='top', fontsize=14, color='black')

    # Add vertical lines between CAH types to separate H1, H2, and H3
    plt.axvline(x=2.5, color='black', linestyle='--')  # Centered between T of H1 and D of H2
    plt.axvline(x=5.5, color='black', linestyle='--')  # Centered between T of H2 and D of H3

    # Add legend for EV charging
    legend_labels = [plt.Line2D([0], [0], color=color_map['Unidirectional'], lw=4), 
                     plt.Line2D([0], [0], color=color_map['Bidirectional'], lw=4)]
    plt.legend(legend_labels, ['Unidirectional', 'Bidirectional'], loc='lower left', fontsize=14, title='EV Charging', title_fontsize='15')

    # Increase the font size of y-ticks
    plt.tick_params(axis='y', labelsize=14)

    # Adjust x-axis limits for more space
    plt.xlim(-0.5, 8.5)  # Add extra space on both sides

    plt.tight_layout()
    plt.show()

# Call the function to create the payback time chart
create_payback_time_chart()
