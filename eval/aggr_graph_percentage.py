import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from percentage reduction CSV files
uni_best_reduction = pd.read_csv('percentage_reduction_uni_best.csv')
uni_worst_reduction = pd.read_csv('percentage_reduction_uni_worst.csv')
bi_best_reduction = pd.read_csv('percentage_reduction_bi_best.csv')
bi_worst_reduction = pd.read_csv('percentage_reduction_bi_worst.csv')

# Custom color palette
custom_colors = ['#88a1cf', '#ff8556', '#a4de52', '#f684c6', '#39c5a3', '#39c5a3']
highlight_color = '#79DAE8'  # Color that pops the most

# Assign specific colors to scenarios
custom_palette = {
    'H+P': custom_colors[0],
    'H+E+P': custom_colors[3],
    'H+E': custom_colors[1],
    'E+P': custom_colors[2],
    'H+P+E+S': custom_colors[4]
}

# Function to plot the graph with simplified legend and adjusted x-axis
def plot_combined_graph(uni_best, uni_worst, bi_best, bi_worst, title):
    plt.figure(figsize=(12, 6))  # Increase figure size for better clarity
    uni_scenarios = uni_best.columns[1:]  # Extract scenario names from uni_best
    bi_scenarios = bi_best.columns[1:]

    combined_scenarios = list(set(uni_scenarios)) 

    for scenario in combined_scenarios:
        color = custom_palette.get(scenario, 'grey')
        
        if scenario == 'H+P':
            # Plot combined H+P scenario lines in a single color
            sns.lineplot(x='Conversion Rate (%)', y='H+P', data=uni_best, 
                         color=color, marker='o', linestyle='-')
            sns.lineplot(x='Conversion Rate (%)', y='H+P', data=uni_worst, 
                         color=color, marker='o', linestyle='--')
            plt.fill_between(x=uni_best['Conversion Rate (%)'], y1=uni_best['H+P'],
                             y2=uni_worst['H+P'], color=color, alpha=0.3)
            # Add combined label
            max_reduction_best = uni_best.loc[uni_best['Conversion Rate (%)'] == uni_best['Conversion Rate (%)'].max(), 'H+P'].values[0]
            plt.text(105, max_reduction_best, 'H + P', color=color, ha='left', va='center', fontsize=14)
        else:
            if scenario in uni_scenarios:
                label = scenario
                if scenario == 'H+E+P':
                    label = 'H + E + P (Uni)'
                if scenario == 'H+E':
                    label = 'H + E (Uni)'
                if scenario == 'E+P':
                    label = 'E + P (Uni)'
                if scenario == 'H+P+E+S':
                    label = 'H + P + E + S (Uni)'

                # Plot unidirectional best scenario lines
                sns.lineplot(x='Conversion Rate (%)', y=scenario, data=uni_best, 
                             color=color, marker='o', linestyle='-')
                # Plot unidirectional worst scenario lines
                sns.lineplot(x='Conversion Rate (%)', y=scenario, data=uni_worst, 
                             color=color, marker='o', linestyle='--')
                plt.fill_between(x=uni_best['Conversion Rate (%)'], y1=uni_best[scenario],
                                 y2=uni_worst[scenario], color=color, alpha=0.3)
                # Adding scenario labels to the right of the lines
                max_reduction_best = uni_best.loc[uni_best['Conversion Rate (%)'] == uni_best['Conversion Rate (%)'].max(), scenario].values[0]
                plt.text(105, max_reduction_best, label, color=color, ha='left', va='center', fontsize=12)

            if scenario in bi_scenarios and scenario != 'H+P+E+S':
                label = scenario
                if scenario == 'H+E+P':
                    label = 'H + E + P (Bi)'
                elif scenario == 'H+E':
                    label = 'H + E (Bi)'
                if scenario == 'E+P':
                    label = 'E + P (Bi)'

                # Plot bidirectional best scenario lines
                sns.lineplot(x='Conversion Rate (%)', y=scenario, data=bi_best, 
                             color=color, marker='o', linestyle='-')
                # Plot bidirectional worst scenario lines
                sns.lineplot(x='Conversion Rate (%)', y=scenario, data=bi_worst, 
                             color=color, marker='o', linestyle='--')
                plt.fill_between(x=bi_best['Conversion Rate (%)'], y1=bi_best[scenario],
                                 y2=bi_worst[scenario], color=color, alpha=0.3)
                # Adding scenario labels to the right of the lines
                max_reduction_best = bi_best.loc[bi_best['Conversion Rate (%)'] == bi_best['Conversion Rate (%)'].max(), scenario].values[0]
                plt.text(105, max_reduction_best, label, color=color, ha='left', va='center', fontsize=12)

    plt.xlabel('Conversion Rate (%)', fontsize=16)  # x-axis label with font size 16
    plt.ylabel('Reduction in CO2 Emissions (%)', fontsize=16)  # y-axis label with font size 16
    plt.xlim([0, 135])  # Extend x-axis to 125 to fit labels
    plt.ylim([80, 0])  # Set y-axis from 100 to 0 to reverse it
    plt.xticks(range(0, 101, 10), fontsize=14)  # Ensure ticks are only up to 100 with font size 14
    plt.yticks(range(0, 80, 10), [f'-{y}' for y in range(0, 80, 10)], fontsize=14)  # Set y-ticks as negative values

    plt.show()

# Plotting the graphs for unidirectional and bidirectional cases
plot_combined_graph(uni_best_reduction, uni_worst_reduction, bi_best_reduction, bi_worst_reduction, 'Unidirectional vs Bidirectional Case: Percentage Reduction in CO2 Emissions')
