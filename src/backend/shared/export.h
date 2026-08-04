/**
 * @file export.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief A macro that enables us to use -fvisibility=hidden.
 * @version 0.1
 * @date 2026-08-04
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#if defined(__GNUC__) || defined(__clang__)
  #define DLLIB_API __attribute__((visibility("default")))
#else
  #define DLLIB_API
#endif
