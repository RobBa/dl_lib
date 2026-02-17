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
void TensorFunctions::ToZeros(Tensor& t) {
  t.reset(0);
}

void TensorFunctions::ToOnes(Tensor& t) {
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
                                         vector<ftype>&& initValues, 
                                         bool requiresGrad) {
  return make_shared<Tensor>(dims, std::move(initValues), requiresGrad);   
}

shared_ptr<Tensor> TensorFunctions::makeSharedTensor(const vector<tensorDim_t>& dims, 
                                           vector<ftype>&& initValues, 
                                           Device d, 
                                           bool requiresGrad){
  return make_shared<Tensor>(dims, std::move(initValues), d, requiresGrad);   
}