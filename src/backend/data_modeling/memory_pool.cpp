/**
 * @file memory_pool.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "memory_pool.h"

#include <utility>
#include <stdexcept>

#ifdef __CUDA
#include "cuda_runtime.h"
#include "utility/cuda/cuda_common.cuh"
#endif

using namespace std;

MemoryPool::~MemoryPool() noexcept {
  flush(Device::CPU);

  #ifdef __CUDA
  flush(Device::CUDA);
  #endif
}

ftype* MemoryPool::allocate(const tensorSize_t size, const Device d) {
  switch(d) {
    case Device::CPU:
    {
      auto& list = freeLists[size];
      if(!list.empty()) {
        ftype* ptr = list.back();
        list.pop_back();
        return ptr;
      }

      ftype* ptr = static_cast<ftype*>(std::malloc(size * sizeof(ftype)));
      if(ptr == nullptr) {
        // ran out of memory, free up cached memory and retry
        flush(Device::CPU);
        ptr = static_cast<ftype*>(std::malloc(size * sizeof(ftype)));
        if(ptr == nullptr) {
          __throw_bad_alloc();
        }
      }

      return ptr;
    }
    case Device::CUDA:
    {
      #ifdef __CUDA
        auto& list = freeListsCuda[size];
        if(!list.empty()) {
          ftype* ptr = list.back();
          list.pop_back();
          return ptr;
        }

        ftype* ptr;
        auto err = cudaMalloc((void**) &ptr, size * sizeof(ftype));
        if(err != cudaSuccess) {
          // ran out of memory, free up cached memory and retry
          flush(Device::CUDA);
          cudaErrchk(cudaMalloc((void**) &ptr, size * sizeof(ftype)));
        }

        return ptr;
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
    }
  }

}
    
void MemoryPool::deallocate(ftype* ptr, const Device d, const tensorSize_t size) {
  switch(d) {
    case Device::CPU:
      freeLists[size].push_back(ptr); 
      break;
    case Device::CUDA:
      #ifdef __CUDA
        freeListsCuda[size].push_back(ptr); 
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }
}

void MemoryPool::flush(Device d) noexcept {
  switch(d) {
    case Device::CPU:
    {
      for (auto& [size, ptrs] : freeLists) {
        for (auto* ptr : ptrs) {
          free(ptr);
        }
      }
      break;
    }
    case Device::CUDA:
    {
      #ifdef __CUDA
        for (auto& [size, ptrs] : freeListsCuda) {
          for (auto* ptr : ptrs) {    
            cudaFree(ptr);
          }
        }
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
    }
  }


}