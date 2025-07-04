import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV files
uni_best = pd.read_csv('1707_e_uni_best_all_scenarios.csv')
uni_worst = pd.read_csv('1707_e_uni_worst_all_scenarios.csv')
bi_best = pd.read_csv('1707_e_bi_best_all_scenarios.csv')
bi_worst = pd.read_csv('1707_e_bi_worst_all_scenarios.csv')

# Function to convert emission values from kilotonnes to megatonnes
def convert_to_megatonnes(df):
    df.iloc[:, 1:] = df.iloc[:, 1:] / 1000
    return df

# Convert all dataframes
uni_best = convert_to_megatonnes(uni_best)
uni_worst = convert_to_megatonnes(uni_worst)
bi_best = convert_to_megatonnes(bi_best)
bi_worst = convert_to_megatonnes(bi_worst)

# Function to plot the graph with simplified legend and adjusted x-axis
def plot_graph(df_best, df_worst, title):
    plt.figure(figsize=(12, 6))  # Increase figure size for better clarity
    scenarios = df_best.columns[1:]  # assuming first column is 'Conversion Rate (%)'
    palette = sns.color_palette("husl", len(scenarios))  # Define color palette

    # Initialize an empty list for custom legend handles
    legend_handles = []

    for i, scenario in enumerate(scenarios):
        scenario_name = scenario.split(' ')[0]  # Clean up scenario names if needed

        # Plot best scenario lines
        sns.lineplot(x='Conversion Rate (%)', y=scenario, data=df_best, 
                     color=palette[i], marker='o', linestyle='-')
        
        # Plot worst scenario lines
        sns.lineplot(x='Conversion Rate (%)', y=scenario, data=df_worst, 
                     color=palette[i], marker='o', linestyle='--')
        
        # Filling the area between the lines
        plt.fill_between(x=df_best['Conversion Rate (%)'], y1=df_best[scenario],
                         y2=df_worst[scenario], color=palette[i], alpha=0.3)

        # Add custom legend handles
        legend_handles.append(plt.Line2D([0], [0], color=palette[i], label=scenario_name))

        # Adding scenario labels to the right of the lines
        max_emission_best = df_best.loc[df_best['Conversion Rate (%)'] == df_best['Conversion Rate (%)'].max(), scenario].values[0]
        plt.text(105, max_emission_best, scenario_name, color=palette[i], ha='left', va='center')

    plt.title(title)
    plt.xlabel('Conversion Rate (%)')
    plt.ylabel('Total CO2 Emissions (Megatonnes)')
    plt.xlim([0, 125])  # Extend x-axis to 110 to fit labels, but no ticks beyond 100
    plt.ylim([6, 20])  # Adjust according to your data range (6000 to 15000 kilotonnes -> 6 to 15 megatonnes)
    plt.xticks(range(0, 101, 10))  # Ensure ticks are only up to 100
    
    # Apply custom legend
    plt.legend(handles=legend_handles)
    
    plt.show()

# Plotting the graphs for unidirectional and bidirectional cases
plot_graph(uni_best, uni_worst, 'Unidirectional Case: CO2 Emissions vs. Conversion Rate')
plot_graph(bi_best, bi_worst, 'Bidirectional Case: CO2 Emissions vs. Conversion Rate')
