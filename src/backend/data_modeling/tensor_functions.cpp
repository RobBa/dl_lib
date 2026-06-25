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

#ifdef __CUDA
#include "data_modeling/cuda/tensor_ops.cuh"
#include "computational_graph/tensor_ops/cuda/tensor_ops_nodes.cuh"
#endif

using namespace std;

Tensor TensorFunctions::Zeros(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(0.0f);
  return res;
}
    
Tensor TensorFunctions::Zeros(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Zeros(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

Tensor TensorFunctions::Ones(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(1.0f);
  return res;
}
    
Tensor TensorFunctions::Ones(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Ones(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

Tensor TensorFunctions::Gaussian(vector<tensorDim_t> dims, const ftype stddev,  
                                 const Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(std::make_shared<utility::GaussianInitializer>(stddev));
  return res;
}
    
Tensor TensorFunctions::Gaussian(vector<tensorDim_t> dims, const ftype stddev, 
                                 const bool requiresGrad) {
  return Gaussian(std::move(dims), stddev, Tensor::getDefaultDevice(), requiresGrad);
}

// Tensor manipulation
void TensorFunctions::ToZeros(Tensor& t) noexcept {
  t.reset(0.0f);
}

void TensorFunctions::ToOnes(Tensor& t) noexcept {
  t.reset(1.0f);
}

void TensorFunctions::ToGaussian(Tensor& t, const ftype stddev) {
  t.reset(std::make_shared<utility::GaussianInitializer>(stddev));
}

shared_ptr<Tensor> TensorFunctions::makeSharedTensor(const vector<tensorDim_t>& dims, bool requiresGrad){
  return make_shared<Tensor>(dims, requiresGrad);   
}

shared_ptr<Tensor> TensorFunctions::makeSharedTensor(const vector<tensorDim_t>& dims, Device d, bool requiresGrad){
  return make_shared<Tensor>(dims, d, requiresGrad);   
}

shared_ptr<Tensor> TensorFunctions::makeSharedTensor(const vector<tensorDim_t>& dims, 
                                         const vector<ftype>& initValues, 
                                         bool requiresGrad) {
  return make_shared<Tensor>(dims, initValues, requiresGrad);
}

shared_ptr<Tensor> TensorFunctions::makeSharedTensor(const vector<tensorDim_t>& dims, 
                                           const vector<ftype>& initValues, 
                                           Device d, 
                                           bool requiresGrad){
  return make_shared<Tensor>(dims, initValues, d, requiresGrad);   
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
Tensor TensorFunctions::SumOverDims(const Tensor& t, tensorDim_t dim) {
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
            
            res.set(res.get(dstIdx) + t.get(srcIdx), dstIdx);
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