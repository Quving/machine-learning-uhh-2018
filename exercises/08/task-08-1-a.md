
### Question:
Why are there (seven) different types (bsr matrix . . . lil matrix) of sparse matrices? Explain the memory layout of csr matrix.


### Answer:
Sparse matrices are used combinatorics and application areas such as network theory, which have a low density of significant data or connections.
Using specialized algorithms and data structures for storing and manipulating is more beneficial regarding memory usage than using the standard sparse matrix structure.

#### CSR-Matrices:
The CSR format stores a sparse m × n matrix M in row form using three (one-dimensional) arrays (A, IA, JA)
- IA[0] = 0
- IA[i] = IA[i − 1] + (number of nonzero elements on the (i-1)-th row in the original matrix)
- The third array, JA, contains the column index in M of each element of A and hence is of length NNZ as well.

#### Example
**Given a 4x4 Matrix:**

0 | 0 | 0 | 0
--- | --- | --- | ---
0 | 5 | 8 | 0
0 | 0 | 3 | 0
0 | 6 | 0 | 0


**In CSR-Matrix:**
   - A  = [ 5 8 3 6 ]
   - IA = [ 0 0 2 3 4 ]
   - JA = [ 0 1 2 1 ]


