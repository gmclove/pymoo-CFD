import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('dump/ics_temporals.txt',skip_header = 1)
noffset = 8*data.shape[0]//10
plt.plot(data[noffset:,3], data[noffset:,4] - data[noffset:,6])
