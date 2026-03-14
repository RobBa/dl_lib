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

using namespace std;

Tensor TensorFunctions::Zeros(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(0);
  return res;
}
    
Tensor TensorFunctions::Zeros(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Zeros(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

Tensor TensorFunctions::Ones(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(1);
  return res;
}
    
Tensor TensorFunctions::Ones(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Ones(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

Tensor TensorFunctions::Gaussian(vector<tensorDim_t> dims, Device d, const bool requiresGrad) {
  auto res = Tensor(std::move(dims), d, requiresGrad);
  res.reset(utility::InitClass::Gaussian);
  return res;
}
    
Tensor TensorFunctions::Gaussian(vector<tensorDim_t> dims, const bool requiresGrad) {
  return Gaussian(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
}

// Tensor manipulation
void TensorFunctions::ToZeros(Tensor& t) noexcept {
  t.reset(0);
}

void TensorFunctions::ToOnes(Tensor& t) noexcept {
  t.reset(1);
}

void TensorFunctions::ToGaussian(Tensor& t) {
  t.reset(utility::InitClass::Gaussian);
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
  if(dim>=t.getDims().nDims()-1){
    __throw_invalid_argument("Dim parameter must be smaller than number of dims, but was " + dim);
  }

  auto resDims = t.getDims().collapseDimension(dim);
  Tensor res = Zeros(resDims.toVector(), t.getDevice(), t.getRequiresGrad()); // inefficiency toVector

  tensorSize_t stride = 1;
  for(tensorDim_t i=dim+1; i<t.getDims().nDims(); i++){
    stride *= t.getDims().getItem(i);
  }
  
  tensorSize_t targetOffset = 0;
  for(tensorDim_t loop=0; loop<t.getDims().getItem(dim); loop++){
    for(tensorSize_t i=0; i<stride; i++){
      res.setItem(res.getItem(i) + t.getItem(targetOffset), i);
      targetOffset++;
    }
  }
  
  return res;
}