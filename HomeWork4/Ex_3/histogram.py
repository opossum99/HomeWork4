import numpy as np
import matplotlib.pyplot as plt

histogram = np.loadtxt("histogram.txt")

# Plot histogram
plt.bar(range(256), histogram, width=1.0, edgecolor='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Grayscale Image')
plt.tight_layout()
plt.savefig('histogram.png', dpi=300)