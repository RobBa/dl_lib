/**
 * @file device.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

enum class Device {
    CPU,
    CUDA
};

const char* DeviceToString(Device d);