/**
 * @file dim_type.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-01-18
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "dim_type.h"

using namespace std;

Dimension::Dimension(vector<tensorDim_t>&& dims, dim_t&& strides)
  : contiguousDims{make_shared<vector<tensorDim_t>>(dims)},
    contiguousStrides{make_shared<dim_t>(makeStrides(dims))},
    dims{dims},
    strides{*contiguousStrides}
{
  size = multVector(dims);
  lastDimIdx = dims.size() - 1;
  assert(size > 0);
}

/**
 * @brief This method gets interesting when we want to get a copy of
 * this dimension instance, but we collapsed one of the dimensions.
 * E.g. when we have a tensor, and we sum over one of its dimensions
 * to get a new tensor, then this will be the new dimensions of the result.
 *
 * Example: t=Tensor with dims (b-size, d). We sum over all batches and
 * get a new tensor tSum=Tensor with dims (d).
 *
 * @param idx The dimension to collapse.
 */
Dimension Dimension::collapseDimension(int idx) const {
  auto mappedIdx = get(idx);

  vector<tensorDim_t> newDims;
  newDims.reserve(dims.size() - 1);
  newDims.insert(newDims.end(), dims.begin(), dims.begin() + idx);
  newDims.insert(newDims.end(), dims.begin() + idx + 1, dims.end());

  dim_t newStrides{};
  tensorDim_t strideIdx = 0;
  for(tensorDim_t i = 0; i < strides.size(); i++){
    if(i==mappedIdx)
      continue;

    newStrides[strideIdx] = strides[i];
  }

  return Dimension(move(newDims), move(newStrides));
}

/**
 * @brief For printouts.
 */
ostream& operator<<(ostream& os, const Dimension& d) noexcept {
  if(d.size > 0){
    os << "\n(";
    for(int i=0; i<d.nDims(); i++){
      os << d.get(i);

      if(i+1 < d.nDims()){
        os << ",";
      }
    }
    os << ")";
    return os;
  }

  os << "\nempty";
  return os;
}
