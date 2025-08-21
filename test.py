import numpy as np
import csv

# load = np.loadtxt("solothurn10.txt")
# print(np.sum(load))

# load = np.loadtxt("./out_noshading/Group_1_Meshes_1_hourly.txt")
# print(np.sum(load))

# solar = np.loadtxt("./out/Group_56_Meshes_111-112_hourly.txt")
# print(np.sum(solar))

# solar = np.loadtxt("./out/Group_50_Meshes_99-100_hourly.txt")
# print(np.sum(solar))
# print(np.sum(solar)/365)
# print(np.sum(solar)/365*5)

# solar = [i*5 for i in solar]
# np.savetxt("solar50.txt", solar)

# solar = np.loadtxt("./out/Group_1_Meshes_1-2_hourly.txt")
# print(np.sum(solar))
# print(np.sum(solar)/365)
# print(np.sum(solar)/365*5)

# solar = [i*5 for i in solar]
# np.savetxt("solar1.txt", solar)

# solar = np.loadtxt("solar.txt")
# print(np.sum(solar))

import pandas as pd


solothurn = pd.read_csv("ontario.csv")
ac = solothurn["AC System Output (W)"]
with open(f'ontario_155_8.txt', 'w', newline='') as f:
    wr = csv.writer(f, delimiter=',')
    for value in ac:
        wr.writerow([value / 1000.0])
