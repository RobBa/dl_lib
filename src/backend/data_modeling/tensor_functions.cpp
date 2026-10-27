/**
 * @file tensor_functions.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-31
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "tensor_functions.h"
#include "shared/initializers.h"

#include <cstring>

#ifdef __CUDA
#include "data_modeling/cuda/tensor_ops.cuh"
#include "computational_graph/tensor_ops/cuda/tensor_ops_nodes.cuh"
#endif

using namespace std;

Tensor tensorfunct::Zeros(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(0.0f);
  return res;
}
    
Tensor tensorfunct::Zeros(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Zeros(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

Tensor tensorfunct::Ones(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(1.0f);
  return res;
}
    
Tensor tensorfunct::Ones(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Ones(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

Tensor tensorfunct::Gaussian(vector<tensorDim_t> dims, const ftype stddev,  
                                 const Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(std::make_shared<utility::GaussianInitializer>(stddev));
  return res;
}
    
Tensor tensorfunct::Gaussian(vector<tensorDim_t> dims, const ftype stddev, 
                                 const bool requiresGrad) {
  return Gaussian(std::move(dims), stddev, Tensor::getDefaultDevice(), requiresGrad);
}

shared_ptr<Tensor> tensorfunct::makeSharedTensor(const vector<tensorDim_t>& dims, bool requiresGrad){
  return make_shared<Tensor>(dims, requiresGrad);   
}

shared_ptr<Tensor> tensorfunct::makeSharedTensor(const vector<tensorDim_t>& dims, Device d, bool requiresGrad){
  return make_shared<Tensor>(dims, d, requiresGrad);   
}

shared_ptr<Tensor> tensorfunct::makeSharedTensor(const vector<tensorDim_t>& dims, 
                                         const vector<ftype>& initValues, 
                                         bool requiresGrad) {
  return make_shared<Tensor>(dims, initValues, requiresGrad);
}

shared_ptr<Tensor> tensorfunct::makeSharedTensor(const vector<tensorDim_t>& dims, 
                                           const vector<ftype>& initValues, 
                                           Device d, 
                                           bool requiresGrad){
  return make_shared<Tensor>(dims, initValues, d, requiresGrad);   
}

/************************************************************************************
 ******************************** Tensor manipulation *******************************
 ***********************************************************************************/

namespace {
  // For the two concats. asserts that the dimensionality requirements described in the description of the functions
  void assertDims(const std::vector<tensorDim_t>& left, const Dimension& right, const int dim) {
    if(left.size() != right.nDims()) {
      throw std::invalid_argument("Dimensions do not match in concat");
    }

    for(int i = 0; i < left.size(); i++) {
      if(i == dim) {
        continue;
      }
      else if(left[i] != right.get(i)) [[unlikely]] {
        throw std::invalid_argument("Dimensions do not match in concat");
      }
    }
  };

  /**
   * @brief Copies the contents of source tensor into target, as specified by the dimension.
   * 
   * @param target The target where we copy into, i.e. the concatenated tensor.
   * @param source The current source to be concatenated against target.
   * @param dim The dimension which we want to concatenate.
   * @param subArrSize The size of the sub-array in source that we want to copy.
   * @param targetDimOffset Where in the concatenated dim do we copy the values of source?
   * 
   * Understanding targetDimOffset: Say we concatenate two tensors of dimensionalities (2, 3, 4) and (2, 5, 4) 
   * along dimension 1, which is has dimensionalities 3 and 4. At first we create a target tensor with res dims
   * (2, 8, 4) in the outer calling functions, see concat(). Then we first copy tensor with dims (2, 3, 4), where 
   * targetDimOffset is 0, since we copy every value directly at the start of every offset of dimension 1. Copying
   * the second tensor however we to copy the values into the respective bins. Each bin of dim 1 has 4 values, and 
   * the first tensor has 3 of them, hence the first value of the second tensor goes into index 12 within the second 
   * dimensions of the target tensor. A third concatenated tensor would have a larger offset and so on...
   */
  void copyValues(Tensor& target, const ftype* const srcValues, const tensorSize_t sourceSize, const tensorSize_t dim,
                  const tensorSize_t subArrSize, const tensorSize_t targetDimOffset) {
    assert(dim > 0); // for dim == 0 choose appendValues()

    const auto nCopies = sourceSize / subArrSize;
    const auto targetDimSize = target.getDims().getStride(dim - 1);

    auto tgtValues = target.data();

    switch(target.getDevice()) {
      case Device::CPU:
        for(int i = 0; i < nCopies; i++) {
          std::memcpy(tgtValues + i * targetDimSize + targetDimOffset, srcValues + i * subArrSize, subArrSize * sizeof(ftype));
        }
        break;
      case Device::CUDA:
      #ifdef __CUDA
        for(int i = 0; i < nCopies; i++) {
          cudaErrchk(cudaMemcpy(tgtValues + i * targetDimSize + targetDimOffset, srcValues + i * subArrSize, 
                                subArrSize * sizeof(ftype), cudaMemcpyDeviceToDevice));
        }
      #else 
        throw runtime_error("Received CUDA tensor in non-CUDA compiled version.");
      #endif
        break;
      default:
        __throw_invalid_argument("Encountered unexpected device");
    }
  };

  void appendValues(Tensor& target, const ftype* const srcValues, const tensorSize_t sourceSize, const tensorSize_t targetOffset) {
    auto tgtValues = target.data() + targetOffset;

    switch(target.getDevice()) {
      case Device::CPU:
        std::memcpy(tgtValues, srcValues, sourceSize * sizeof(ftype));
        break;
      case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaMemcpy(tgtValues, srcValues, sourceSize * sizeof(ftype), cudaMemcpyDeviceToDevice));
      #else 
        throw runtime_error("Received CUDA tensor in non-CUDA compiled version.");
      #endif
        break;
      default:
        __throw_invalid_argument("Encountered unexpected device");
    }
  };
}

/**
 * @brief Implementation of the relevant concat()-overloads. 
 * 
 * Two vectors of sizes [dim11, dim12, ..., dim1n] and [dim21, dim22, ..., dim2n] can be 
 * concatenated along dimension dimx only iff pairs [dim11, dim21], ..., [dim1n, dim2n]
 * are the same except dimx (i.e. [dim1x, dim2x] are allowed to be different).
 * 
 * The result has dimensionality [dim11, dim12, ... (dim1x + dim2x + ...), ..., dim1n].
 * 
 * Operation works in three steps: 
 * 1. Determine offset in target array that we need to copy in. 
 * 2. Determine the size of each sub-array we need to copy into result.
 * 3. Copy values into target.
 */
Tensor tensorfunct::concatImpl(std::span<const Tensor* const> input, const tensorDim_t dim) {
  assert(input.size() > 0);

  // compute the dimensions of the result
  vector<tensorDim_t> resDims = input[0]->getDims().toVector();
  for(int i = 1; i < input.size(); i++) {
    const auto& rightDims = input[i]->getDims();
    assertDims(resDims, rightDims, dim);
    resDims[dim] += rightDims.get(dim);
  }

  auto res = Tensor(std::move(resDims), input[0]->getDevice());

  if(dim > 0) [[likely]] {
    tensorSize_t targetDimOffset = 0;
    for(const Tensor* const src : input) {
      assert(res.getDevice() == src->getDevice());

      // the size of each dimension when concatenating
      const auto subArrSize = src->getDims().getStride(dim - 1);

      src->makeContiguous();
      copyValues(res, src->data(), src->getSize(), dim, subArrSize, targetDimOffset);

      targetDimOffset += subArrSize;
    }
  }
  else [[unlikely]] { // simplifies to appending the arrays one after the other
    tensorSize_t tgtOffset = 0; // offset in target that we append to

    for(auto& src : input) {
      assert(res.getDevice() == src->getDevice());

      src->makeContiguous();
      appendValues(res, src->data(), src->getSize(), tgtOffset);
      tgtOffset += src->getSize();
    }
  }

  return res;
}

/**
 * @brief Concatenates the tensors along specified dimension.
 */
Tensor tensorfunct::concat(const std::vector<Tensor>& input, int dim) {
#ifndef NDEBUG
  if(input.size() < 2) {
    throw std::invalid_argument("Concat must have at least two tensors");
  }
#endif

  if(dim < 0) {
    dim += input.at(0).getDims().nDims();
    assert(dim >= 0);
  }

  std::vector<const Tensor*> ptrs;
  ptrs.reserve(input.size());

  for (const auto& t : input) {
    ptrs.push_back(&t);
  }

  return concatImpl(ptrs, dim);
}

void tensorfunct::ToZeros(Tensor& t) noexcept {
  t.reset(0.0f);
}

void tensorfunct::ToOnes(Tensor& t) noexcept {
  t.reset(1.0f);
}

void tensorfunct::ToGaussian(Tensor& t, const ftype stddev) {
  t.reset(std::make_shared<utility::GaussianInitializer>(stddev));
}

/************************************************************************************
 ************************************ Arithmetics ***********************************
 ***********************************************************************************/

 /**
  * @brief Sums over the dimensions. If input is (b-size, dim1, dim2), and 
  * input dim-parameter is 1, then output will be (b-size, dim2). If 
  * input dim-parameter is 0, then output will be (dim1, dim2).
  * Input dim must be smaller then t.dims.nDims()-1
  */
Tensor tensorfunct::SumOverDims(const Tensor& t, tensorDim_t dim) {
  if(dim > t.getDims().nDims() - 1){
    __throw_invalid_argument("Dim parameter must be smaller than number of dims, but was " + dim);
  }

  auto resDims = t.getDims().collapseDimension(dim);
  Tensor res = Zeros(resDims.toVector(), t.getDevice(), t.getRequiresGrad()); // inefficiency toVector

  switch(t.getDevice()) {
    case Device::CPU:
    {
      tensorSize_t stride = 1;
      for(tensorDim_t i = dim + 1; i < t.getDims().nDims(); i++){
        stride *= t.getDims()[i];
      }

      // size of dimensions before dim
      tensorSize_t outerSize = 1;
      for(tensorDim_t i = 0; i < dim; i++) {
        outerSize *= t.getDims()[i];
      }
      
      for(tensorSize_t outer = 0; outer < outerSize; outer++){
        for(tensorDim_t k = 0; k < t.getDims()[dim]; k++){
          for(tensorSize_t i = 0; i < stride; i++){
            tensorSize_t srcIdx = outer * t.getDims()[dim] * stride + k * stride + i;
            tensorSize_t dstIdx = outer * stride + i;
            
            res.data()[dstIdx] = res.data()[dstIdx] + t.data()[srcIdx];
          }
        }
      }
      break;
    }
    case Device::CUDA:
      #ifdef __CUDA
        cuda_impl::sumOverDims(res, t, dim);
      #else
        __throw_invalid_argument("Not compiled with CUDA");
      #endif
      break;
  }
  
  return res;
}