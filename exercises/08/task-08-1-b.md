### Question i)
Which of these classes is best for inserting new non-zero elements into an existing sparse
matrix?

#### Answer
Classes that are typically used for construction of matrices are *DOK* and *LIL*.
- DOK which is the shorten version of Dictionary of Keys maps (row, column) pairs to the value of the elements. Missing values are considered as zeros.
- LIL stores one list per row, with each entry containing the column index and the value.
Hence, I would choose DOK since the order of the insertion isn't weighted as important according to the question.

### Question ii)
Which of these classes are best for matrix-vector dot products and matrix-matrix multiplications? Why?

#### Answer

