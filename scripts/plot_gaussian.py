import matplotlib.pyplot as plt
import numpy as np

# Simulate normally distributed data
data = np.random.normal(0, 1, 10000)

# Set thresholds at 10% from the bottom and top
lower_threshold = np.percentile(data, 10)
upper_threshold = np.percentile(data, 90)

# Create histogram without axis labels, ticks, title, or grid lines
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(data, bins=50, edgecolor="black", alpha=0.7)

# Color the bottom 10% in red and top 10% in green
for i in range(len(patches)):
    if bins[i] < lower_threshold:
        patches[i].set_facecolor("red")
    elif bins[i] > upper_threshold:
        patches[i].set_facecolor("green")
    else:
        patches[i].set_facecolor("blue")

# Remove the axes, ticks, gridlines, and title
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["left"].set_visible(False)
# plt.gca().spines["bottom"].set_visible(False)
plt.grid(False)
plt.title("")

# Show axes, x is "Impact", y is "Labs"
plt.xlabel('"Impact"')
plt.ylabel("Number of Labs")

# Show the simplified plot
plt.show()
