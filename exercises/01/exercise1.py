import numpy as np
import matplotlib.pyplot as plt

traffic = np.genfromtxt('traffic_per_hour.csv')
print(traffic[:,1])

plt.scatter(traffic[:,0], traffic[:,1])
plt.show()
