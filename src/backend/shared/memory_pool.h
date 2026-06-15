/**
 * @file memory_pool.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "shared/global_params.h"
#include "data_modeling/device.h"
#include "utility/utils.h"

#include <unordered_map>
#include <vector>

#include <utility>
#include <stdexcept>

#ifdef __CUDA
#include "cuda_runtime.h"
#include "utility/cuda/cuda_common.cuh"
#endif

namespace mempool_impl {
  template<typename T>
  class MemoryPool final {
    private:
      // hold pointers to unused allocated memory.
      std::unordered_map<tensorSize_t, std::vector<T*>> freeLists;
      std::unordered_map<tensorSize_t, std::vector<T*>> freeListsCuda;
      
    public:
      MemoryPool() = default;
      ~MemoryPool() noexcept;

      T* request(Device d, tensorSize_t n);
      void giveback(T* ptr, Device d, tensorSize_t n);
      void flush(Device d) noexcept;
  };

  /**
   * @brief Destroy the Memory Pool:: Memory Pool
   * 
   * Frees all held memory.
   * 
   */
  template<typename T>
  MemoryPool<T>::~MemoryPool() noexcept {
    MemoryPool<T>::flush(Device::CPU);

    #ifdef __CUDA
    MemoryPool<T>::flush(Device::CUDA);
    #endif
  }

  /**
   * @brief Requests a piece of memory from the memory pool of n elements.
   * If pool has free memory of n elements, then it returns a pointer to that.
   * Otherwise it allocates memory of size n * sizeof(T) and returns the new pointer.
   * 
   * @param n Number of elements requested.
   * @param d Device. Dictates which type of memory is requested.
   * @return T* Pointer to allocated memory.
   */
  template<typename T>
  T* MemoryPool<T>::request(const Device d, const tensorSize_t n) {
    switch(d) {
      case Device::CPU:
      {
        auto& list = freeLists[n];
        if(!list.empty()) {
          T* ptr = list.back();
          ASSERT_HOST_PTR(ptr);
          list.pop_back();
          return ptr;
        }

        T* ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
        if(ptr == nullptr) {
          // ran out of memory, free up cached memory and retry
          flush(Device::CPU);
          ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
          if(ptr == nullptr) {
            std::__throw_bad_alloc();
          }
        }

        ASSERT_HOST_PTR(ptr);
        return ptr;
      }
      case Device::CUDA:
      {
        #ifdef __CUDA
          auto& list = freeListsCuda[n];
          if(!list.empty()) {
            T* ptr = list.back();
            ASSERT_DEVICE_PTR(ptr);

            list.pop_back();
            return ptr;
          }

          T* ptr;
          auto err = cudaMalloc((void**) &ptr, n * sizeof(T));
          if(err != cudaSuccess) {
            // ran out of memory, free up cached memory and retry
            flush(Device::CUDA);
            cudaErrchk(cudaMalloc((void**) &ptr, n * sizeof(T)));
          }

          ASSERT_DEVICE_PTR(ptr);
          return ptr;
        #else 
          std::__throw_runtime_error("Not compiled with CUDA");
        #endif
      }
    }

  }

  /**
   * @brief Returns the piece of memory pointed to by ptr back to the memory pool.
   */
  template<typename T>
  void MemoryPool<T>::giveback(T* ptr, const Device d, const tensorSize_t n) {
    switch(d) {
      case Device::CPU:
        ASSERT_HOST_PTR(ptr);
        freeLists[n].push_back(ptr); 
        break;
      case Device::CUDA:
        #ifdef __CUDA
          ASSERT_DEVICE_PTR(ptr);
          freeListsCuda[n].push_back(ptr); 
        #else 
          std::__throw_runtime_error("Not compiled with CUDA");
        #endif
        break;
    }
  }

  /**
   * @brief Frees all held memory on requested device.
   * 
   * @param d Device whose memory to flush.
   */
  template<typename T>
  void MemoryPool<T>::flush(const Device d) noexcept {
    switch(d) {
      case Device::CPU:
      {
        for (auto& [n, ptrs] : freeLists) {
          for (auto* ptr : ptrs) {
            ASSERT_HOST_PTR(ptr);
            free(ptr);
          }
        }
        break;
      }
      case Device::CUDA:
      {
        #ifdef __CUDA
          for (auto& [n, ptrs] : freeListsCuda) {
            for (auto* ptr : ptrs) {
              ASSERT_DEVICE_PTR(ptr); 
              cudaFree(ptr);
            }
          }
        #else 
          std::__throw_runtime_error("Not compiled with CUDA");
        #endif
        break;
      }
    }
  }
}

namespace mempool {
  inline static mempool_impl::MemoryPool<ftype> tensorPool;
  inline static mempool_impl::MemoryPool<tensorDim_t> tensorDimPool;
}