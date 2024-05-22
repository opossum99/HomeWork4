import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

u = np.loadtxt("out.txt")
hm = sn.heatmap(data=u, vmin=0, vmax=1)

tics = list(range(1,len(u),len(u)//10))
real_ticks = np.linspace(1/len(u),tics[-1]/len(u),len(tics))
plt.xticks(tics, np.around(real_ticks,2))
plt.yticks(tics, np.around(real_ticks,2))
plt.show()
#plt.savefig("heatmap")
