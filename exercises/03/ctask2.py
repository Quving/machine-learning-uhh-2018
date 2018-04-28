from matplotlib import pyplot as plt
from scipy import io as io
import numpy as np
import math

### A
# Load mat data file
loaded_data = io.loadmat('Adot.mat')

# access the arrays inside the data structure
X = loaded_data['X']

# create a numpy matrix for the linear mapping V, where
# theta = pi/3
# V = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
theta = np.pi/3
V = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

### B1
# Apply linear mapping on X to get Y = VX
# Plot both x and y in the same figure
# Question: What does the linear mapping do?

Y = np.linalg.multi_dot([V, X])

plt.figure()

plt.subplot(1,3,1)

plt.title("Assignment 03.2.b1: Linear mapping")
plt.scatter(X[0], X[1], color="blue")
plt.scatter(Y[0], Y[1], color="red")


plt.subplot(1,3,2)

print('''The linear mapping transforms the values of X to a linear distribution Y.''')

Z = np.linalg.multi_dot([np.matrix.transpose(V), Y])

plt.title("Assignment 03.2.b2: Linear mapping")
plt.scatter(Z[0], Z[1], color="green")

plt.title("Overlap of X, Y, Z")
plt.subplot(1,3,3)
plt.scatter(X[0], X[1], color="blue")
plt.scatter(Y[0], Y[1], color="red")
plt.scatter(Z[0], Z[1], color="green")

print('''The result of the application of the transposed V matrix to Y is the same as Y. V transposed is therefore the idendity matrix of Y''')

plt.show()