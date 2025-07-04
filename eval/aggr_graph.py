import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# this creates the green gaphs, call aggreagate_results.py to create the csv files

# Load data from CSV files
uni_best = pd.read_csv('e_uni_best.csv')
uni_worst = pd.read_csv('e_uni_worst.csv')
bi_best = pd.read_csv('e_bi_best.csv')
bi_worst = pd.read_csv('e_bi_worst.csv')

# Function to convert emission values from kilotonnes to megatonnes
def convert_to_megatonnes(df):
    df.iloc[:, 1:] = df.iloc[:, 1:] / 1000000
    return df

# Convert all dataframes
uni_best = convert_to_megatonnes(uni_best)
uni_worst = convert_to_megatonnes(uni_worst)
bi_best = convert_to_megatonnes(bi_best)
bi_worst = convert_to_megatonnes(bi_worst)

# Function to plot the graph with improved annotations
def plot_graph(df_best, df_worst, title, vertical_spacings):
    plt.figure(figsize=(10, 6))
    # Define two shades of green
    green_palette = ["#2ca02c", "#98df8a"]

    # Plotting best and worst scenario lines
    sns.lineplot(x='Conversion Rate (%)', y='Total Emissions T', data=df_best, 
                 label='Best Scenario', marker='o', color=green_palette[0])
    sns.lineplot(x='Conversion Rate (%)', y='Total Emissions T', data=df_worst, 
                 label='Worst Scenario', marker='o', color=green_palette[1])
    
    # Filling the area between the lines
    plt.fill_between(x=df_best['Conversion Rate (%)'], y1=df_best['Total Emissions T'],
                     y2=df_worst['Total Emissions T'], color='gray', alpha=0.3)

    # Adding annotations for selected conversion rates with adjusted distances
    rates = [0, 20, 40, 60, 80, 100]
    for rate in rates:
        best_emission = df_best.loc[df_best['Conversion Rate (%)'] == rate, 'Total Emissions T'].values[0]
        worst_emission = df_worst.loc[df_worst['Conversion Rate (%)'] == rate, 'Total Emissions T'].values[0]
        if rate == 0:
            plt.text(rate, best_emission + vertical_spacings[rate], f'{best_emission:.2f}', va='center', ha='center', fontsize=11, color=green_palette[0])
        else:
            plt.text(rate, best_emission + vertical_spacings[rate], f'{best_emission:.2f}', color=green_palette[0], va='bottom', ha='center', fontsize=11)
            plt.text(rate, worst_emission - vertical_spacings[rate], f'{worst_emission:.2f}', color=green_palette[1], va='top', ha='center', fontsize=11)

    plt.xlabel('Conversion Rate (%)')
    plt.ylabel('Total CO2 Emissions MT')
    plt.title(title)
    plt.legend()
    plt.show()

# Vertical spacings for unidirectional case
uni_vertical_spacings = {0: 0.05, 20: 0.11, 40: 0.15, 60: 0.18, 80: 0.21, 100: 0.22}

# Vertical spacings for bidirectional case (increased)
bi_vertical_spacings = {0: 0.1, 20: 0.4, 40: 0.6, 60: 0.7, 80: 0.8, 100: 0.9}

# Plotting the graphs for unidirectional and bidirectional cases
plot_graph(uni_best, uni_worst, 'Unidirectional Case: CO2 Emissions vs. Conversion Rate', uni_vertical_spacings)
plot_graph(bi_best, bi_worst, 'Bidirectional Case: CO2 Emissions vs. Conversion Rate', bi_vertical_spacings)
