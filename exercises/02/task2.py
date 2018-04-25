#!/usr/bin/python3

import numpy as np


response2d = "Indeed suspicious, the solutions of the check || h_k * x_k -b || should be a vector of zeros.  But through python's rounding it is possible that the equations solutions are not 100 percent precise resulting in very small numbers instead of zero after check."



response2e = "It's oberverable that the condition number increases very fast by increasingly dimension of the hilbert matrix. The condition number associated with the linear equation Ax = b gives a bound on how inaccurate the solution x will be after approximation.  Note that this is before the effects of round-off error are taken into account; conditioning is a property of the matrix, not the algorithm or floating point accuracy of the computer used to solve the corresponding system."


## TASK 2a ###############################################################

# Generate a Hilbert Matrix for a given dimension.
def generateHilberMat(dimension):
    mat = []
    for k in range(0, dimension):
        row = []
        for l in range(0, dimension):
            row.append(1/(k+l+1))
        mat.append(row)
    return mat


## TASK 2b ###############################################################

# Returns the rank of a given matrix.
def calculateMatrixRank(matrix):
    return np.linalg.matrix_rank(matrix)

# Returns the condition number  of a given matrix.
def calculateMatrixCondition(matrix):
    return np.linalg.cond(matrix)


## TASK 2c ###############################################################

def solveXforHilbert(dimension):
    hilmat = generateHilberMat(dimension)
    a = np.array(hilmat)
    b = np.array([1]*dimension)
    x = np.linalg.solve(a,b)
    checkResult = abs(np.subtract(np.dot(hilmat, x), b))
    if np.allclose(np.dot(a, x), b):
        return x
    else:
        return []



if __name__ == "__main__":
    hmat = generateHilberMat(3) # Task 2a
    for row in hmat:
        print(row)

    for k in [1,30]: # Task 2b
        hilmat = generateHilberMat(k)
        rank = calculateMatrixRank(hilmat)
        cond = calculateMatrixCondition(hilmat)
        print("Hilbert-Matrix of dimension", k, "has a rank of", rank, "and cond of", cond)

    for k in [1,2,3,4,10,15,20,30,50,100]: # Task 2c
        print("\nk:", k)
        print("x: ", solveXforHilbert(k))

    print("\n", response2e) # Task 2e

    print("\n", response2d) # Task 2d
