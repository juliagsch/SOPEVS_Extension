import pandas as pd

# Load data from CSV files
uni_best = pd.read_csv('2507_e_uni_best_all_scenarios.csv')
uni_worst = pd.read_csv('2507_e_uni_worst_all_scenarios.csv')
bi_best = pd.read_csv('2507_e_bi_best_all_scenarios.csv')
bi_worst = pd.read_csv('2507_e_bi_worst_all_scenarios.csv')

# Baseline emission value
baseline_emission = 19130

# Function to calculate percentage reduction compared to the baseline
def calculate_percentage_reduction(df, baseline):
    percentage_reduction = df.copy()
    percentage_reduction.iloc[:, 1:] = (baseline - df.iloc[:, 1:]) / baseline * 100
    return percentage_reduction

# Calculate percentage reductions and save each to a separate CSV file
uni_best_reduction = calculate_percentage_reduction(uni_best, baseline_emission)
uni_best_reduction.to_csv('percentage_reduction_uni_best.csv', index=False)

uni_worst_reduction = calculate_percentage_reduction(uni_worst, baseline_emission)
uni_worst_reduction.to_csv('percentage_reduction_uni_worst.csv', index=False)

bi_best_reduction = calculate_percentage_reduction(bi_best, baseline_emission)
bi_best_reduction.to_csv('percentage_reduction_bi_best.csv', index=False)

bi_worst_reduction = calculate_percentage_reduction(bi_worst, baseline_emission)
bi_worst_reduction.to_csv('percentage_reduction_bi_worst.csv', index=False)

print("Percentage reduction files saved for uni_best, uni_worst, bi_best, and bi_worst.")
