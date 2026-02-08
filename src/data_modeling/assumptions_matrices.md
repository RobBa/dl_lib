# Indexing

In this project, we flatten out our arrays. We assume a maximum number of dimensions 
for a tensor. The challenge then is to find for each kind of tensor, e.g. 
1D, 2D, 3D, ..., to address the right indexes. For this we need to have conventions 
of how they are indexed. Described in the following:

## Scalar and 1D

Trivial

## 2D and larger

We assume the last two dimensions to represent rows and columns. 
In dimensions of order higher than 2 the first dimension is always assumed
to be the batchsize. So a 2D matrix would be something like (nrows, ncols), a 
3D matrix something like (nbatches, nrows, ncols), a 4D matrix something like 
(nbatches, timestep, nrows, ncols), ...

In memory, we do have rows encapsulated by columns and so on. 
2D matrix: 

[[row1]
 [row2]
 ...
 [rown]]

The lengths of row1 through rown are equal, therefore giving us the number of colums.
Higher orders always do encapsulate from right to left, in this order. E.g. a 3D tensor:

[
  // batch 1 starts here
  [[row1]
  [row2]
  ...
  [rown]]

  // batch 2 starts here
  [[row1]
  [row2]
  ...
  [rown]]

  // ...
]

For 4D tensors, sticking with the example above, the timestep encloses the individual 
2D matrices. Then the batches enclose the timesteps, and so forth.


# Matmul

In case of multiplication, the following cases can happen: 

1. At least one scalar tensor: We broadcast
2. Dimensions match, e.g. (b1, b2, m, n) @ (b1, b2, n, p): We multiply all 2 matrices contained -> (b1, b2, m, p)
3. One of the two tensors has one more dimension than the other: We assume batch and broadcast: (m, n) @ (batch, n, p) -> (batch, m, p), (batch, m, n) @ (n, p) -> (batch, m, p)