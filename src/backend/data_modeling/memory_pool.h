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
#include "device.h"

#include <unordered_map>
#include <vector>

class MemoryPool final {
  private:
    std::unordered_map<tensorSize_t, std::vector<ftype*>> freeLists;
    std::unordered_map<tensorSize_t, std::vector<ftype*>> freeListsCuda;
    
  public:
    MemoryPool() = default;
    ~MemoryPool() noexcept;

    ftype* allocate(tensorSize_t size, Device d);
    void deallocate(ftype* ptr, Device d, tensorSize_t size);
    void flush(Device d) noexcept;
};