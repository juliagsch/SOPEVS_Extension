import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'independence.csv'  # Update this path as necessary
data = pd.read_csv(file_path)

# Adjust 'Operation' to policy categories
data['Operation Policy'] = data['Operation'].apply(lambda x: 'Unidirectional' if 'unidirectional' in x.lower() else 'Bidirectional')
data['Operation Policy'] = pd.Categorical(data['Operation Policy'], ['Unidirectional', 'Bidirectional'], ordered=True)

# Define archetypes and CAH types
archetype_order = ['Detached', 'Semi-Detached', 'Terraced']  # D, SD, T
cah_types = ['H1', 'H2', 'H3']
color_map = {'Unidirectional': '#D5F0C1', 'Bidirectional': '#80BCBD'}

# Function to create a combined independence level chart
def create_combined_chart():
    plt.figure(figsize=(14, 8))  # Set figure size

    position = 0  # Track the current x-axis position
    xticks = []   # To store x positions for the ticks
    xtick_labels = []  # To store labels like 'D', 'SD', 'T'

    # Loop through each CAH type and archetype
    for cah in cah_types:
        for archetype in archetype_order:
            # Track if we have plotted for this archetype/cah combination
            position_assigned = False

            # First loop for Unidirectional
            filtered_uni = data[(data['CAH_Type'] == cah) & 
                                (data['Operation Policy'] == 'Unidirectional') & 
                                (data['Archetype'] == archetype)]
            if not filtered_uni.empty:
                independence_l = filtered_uni.iloc[0]['Independence_l']
                independence_h = filtered_uni.iloc[0]['Independence_h']
                range_value = independence_h - independence_l

                # Draw rectangle for Unidirectional
                plt.bar(position, height=range_value, bottom=independence_l, width=0.4,
                        color=color_map['Unidirectional'], alpha=0.7)
                
                position_assigned = True  # Mark that a bar was placed at this position

            # Second loop for Bidirectional, at the same x position
            filtered_bi = data[(data['CAH_Type'] == cah) & 
                               (data['Operation Policy'] == 'Bidirectional') & 
                               (data['Archetype'] == archetype)]
            if not filtered_bi.empty:
                independence_l = filtered_bi.iloc[0]['Independence_l']
                independence_h = filtered_bi.iloc[0]['Independence_h']
                range_value = independence_h - independence_l

                # Draw rectangle for Bidirectional
                plt.bar(position, height=range_value, bottom=independence_l, width=0.4,
                        color=color_map['Bidirectional'], alpha=0.7)

            # Update x-ticks if a bar was assigned
            if position_assigned:
                xticks.append(position)
                xtick_labels.append(archetype[0])  # Use 'D', 'SD', or 'T' as label
                position += 1

        # Add a gap after each archetype for separation
        position += 1

    # Set x-ticks and labels
    plt.xticks(ticks=xticks, labels=xtick_labels, rotation=0, fontsize=14)
    plt.ylabel('Independence (%)', fontsize=16)  # Y-axis label
    plt.xlabel('Archetype', fontsize=16)         # X-axis label
    plt.ylim(0, 100)  # Assuming independence is expressed as a percentage

    # Center CAH labels above the plot
    plt.text(1.0, 80, 'CAH1', ha='center', va='top', fontsize=14, color='black')
    plt.text(5.0, 80, 'CAH2', ha='center', va='top', fontsize=14, color='black')
    plt.text(9.0, 80, 'CAH3', ha='center', va='top', fontsize=14, color='black')

    # Add vertical lines to separate each CAH type
    plt.axvline(x=3, color='black', linestyle='--')
    plt.axvline(x=7, color='black', linestyle='--')

    # Add legend for EV charging
    legend_labels = [plt.Line2D([0], [0], color=color_map['Unidirectional'], lw=4),
                     plt.Line2D([0], [0], color=color_map['Bidirectional'], lw=4)]
    plt.legend(legend_labels, ['Unidirectional', 'Bidirectional'], loc='upper left', fontsize=14, title='EV Charging', title_fontsize='15')

    # Increase the font size of y-ticks
    plt.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.show()

# Call the function to create the combined chart
create_combined_chart()
