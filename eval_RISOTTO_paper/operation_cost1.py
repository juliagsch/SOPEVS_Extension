import pandas as pd

# Load CSV file
df = pd.read_csv('path_to_your_csv_file.csv')

# Combine date and time into a single datetime column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time (US Central Time)'])

# Filter rows for full hours only on the specified date
df = df[(df['DateTime'].dt.date == pd.to_datetime('2024-01-28').date()) & (df['DateTime'].dt.minute == &#8203;``【oaicite:0】``&#8203;
