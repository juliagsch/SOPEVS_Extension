import matplotlib.pyplot as plt
import numpy as np

# Generate example data
hours = np.arange(0, 35001)

fig, ax3 = plt.subplots(figsize=(15, 3))

# EV Usage plot (only showing shaded areas)
ax3.set_ylabel('EV usage (kW)', fontsize=14)
ax3.set_xlabel('Hour', fontsize=14)
# ax3.set_title('EV Usage', fontsize=16)

# Highlight times when EV is at home
home_times = [(0, 10), (13, 16), (18, 32)]
for start, end in home_times:
    ax3.axvspan(start*1000, end*1000, color='lightblue', alpha=0.5)

# Adjust y-limits and add a zero baseline for better visualization
ax3.set_ylim(0, 1)
ax3.set_yticks([])  # Hide y-axis ticks as there is no data to show

# Remove vertical grid lines
ax3.grid(False)

# Adjust the size of the tick labels
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()
