# How indexing works here

In this project, we flatten out our arrays. We assume a maximum number of dimensions 
for a tensor. The challenge then is to find for each kind of tensor, e.g. 
1D, 2D, 3D, ..., to address the right indexes. For this we need to have conventions 
of how they are indexed. Described in the following:

## Scalar and 1D

Trivial

## 2D

We assume row first, then column. For our flat array this means the following order:

[row1, row2, ..., rowN]

, where each row has n_columns entries.

## 3D

Not yet implemented

## 4D

Not yet implemented