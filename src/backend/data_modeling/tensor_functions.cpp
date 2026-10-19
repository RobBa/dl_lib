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
  auto assertDims = [](const std::vector<tensorDim_t>& left, const Dimension& right, const int dim) {
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

  // copies the contents of dst tensor into target
  auto copyValues = [](Tensor& target, const Tensor& source, const tensorSize_t idx) {
    assert(target.getDevice() == source.getDevice());

    auto tgtValues = target.data() + idx;
    const auto srcValues = source.data();

    switch(source.getDevice()) {
      case Device::CPU:
        std::memcpy(tgtValues, srcValues, source.getSize() * sizeof(ftype));
        break;
      case Device::CUDA:
      #ifdef __CUDA__
        cudaErrchk(cudaMemcpy(tgtValues, srcValues, source.getSize() * sizeof(ftype), cudaMemcpyDeviceToDevice));
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
 * @brief Concatenates the tensors along specified dimension.
 * 
 * Two vectors of sizes [dim11, dim12, ..., dim1n] and [dim21, dim22, ..., dim2n] can be 
 * concatenated along dimension dimx only iff pairs [dim11, dim21], ..., [dim1n, dim2n]
 * are the same except dimx (i.e. [dim1x, dim2x] are allowed to be different).
 * 
 * The result has dimensionality [dim11, dim12, ... (dim1x + dim2x), ..., dim1n].
 */
Tensor tensorfunct::concat(const std::vector<Tensor>& input, int dim) {
  assert(input.size() > 0);

  vector<tensorDim_t> resDims = input[0].getDims().toVector();

  if(dim < 0) {
    dim += resDims.size();
  }

  for(int i = 1; i < input.size(); i++) {
    const auto& rightDims = input[i].getDims();
    assertDims(resDims, rightDims, dim);
    resDims[dim] += rightDims.get(dim);
  }

  auto res = Tensor(std::move(resDims), input[0].getDevice());
  tensorSize_t currSize = 0;
  for(const auto& t : input) {
    copyValues(res, t, currSize);
    currSize += t.getSize();
  }

  return res;
}

/**
 * @brief Concatenates the tensors along specified dimension. Autograd possible.
 */
std::shared_ptr<Tensor> tensorfunct::concat(const std::vector< const std::shared_ptr<Tensor> >& input, int dim) {
  assert(input.size() > 0);

  vector<tensorDim_t> resDims = input[0]->getDims().toVector();

  if(dim < 0) {
    dim += resDims.size();
  }

  for(int i = 1; i < input.size(); i++) {
    const auto& rightDims = input[i]->getDims();
    assertDims(resDims, rightDims, dim);
    resDims[dim] += rightDims.get(dim);
  }

  auto res = make_shared<Tensor>(std::move(resDims), input[0]->getDevice());
  tensorSize_t currSize = 0;
  for(const auto& t : input) {
    copyValues(*res, *t, currSize);
    currSize += t->getSize();
  }

  return res;
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