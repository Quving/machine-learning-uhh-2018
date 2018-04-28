from matplotlib import pyplot as plt
from scipy import io as io
import numpy as np
import math

plt.figure()

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

Y = np.linalg.multi_dot([V, X])

# Plot both x and y in the same figure
plt.subplot(2,3,1)

plt.title("Assignment 03.2.b1: Linear mapping")
plt.scatter(X[0], X[1], color="blue")
plt.scatter(Y[0], Y[1], color="red")


plt.subplot(2,3,2)

# Question: What does the linear mapping do?
print('''The linear mapping transforms the values of X.''')


### B2
# Apply V^T to Y

Z = np.linalg.multi_dot([np.matrix.transpose(V), Y])

plt.title("Transposed V")
plt.scatter(Z[0], Z[1], color="green")

# Plot the data
plt.subplot(2,3,3)

plt.title("Overlap of X, Y, Z")
plt.scatter(X[0], X[1], color="blue")
plt.scatter(Y[0], Y[1], color="red")
plt.scatter(Z[0], Z[1], color="green")

# What happened?
print('''The result of the application of the transposed V matrix to Y is the same as X.''')

### C
# Apply linear Mappings D1 and D2 to X
D1 = np.array([[2,0],[0,2]])
D2 = np.array([[2,0],[0,1]])

Y_D1 = np.linalg.multi_dot([D1, Y])
Y_D2 = np.linalg.multi_dot([D2, Y])

# Plot the data
plt.subplot(2,3,4)

plt.title("Overlap of X, Y_D1, Y_D2")

plt.scatter(X[0], X[1], color="blue")
plt.scatter(Y_D1[0], Y_D1[1], color="c")
plt.scatter(Y_D2[0], Y_D2[1], color="m")

# What happened?
print('''D1 (cyan) scales the data uniform by 2. D2 (magenta) scales all values of the first array by 2.''')

### D
# Apply the linear mapping A = V^T * D2 * V
A = np.linalg.multi_dot([np.matrix.transpose(V), D2, V])

# Plot the data
plt.subplot(2,3,5)

plt.title("Apply A = V^T * D2 * V")

plt.scatter(V[0], V[1], color="c")
plt.scatter(D2[0], D2[1], color="m")
plt.scatter(A[0], A[1], color="k")

# What happened?
print('''The linear mapping A gets transformed by the applied matrices''')

plt.show()