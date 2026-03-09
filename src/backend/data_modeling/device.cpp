/**
 * @file device.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "device.h"

#include <stdexcept>

const char* DeviceToString(Device d) {
    switch(d){
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
    }

    std::__throw_invalid_argument("Unknown device encountered");
    return ""; // suppress
}