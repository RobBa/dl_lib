/**
 * @file macros.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-05-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once 

#ifdef NDEBUG
  #define assert_debug(cond, msg) ((void)0)
#else
  #define assert_debug(cond, msg) \
    do { \
      if (!(cond)) { \
        fprintf(stderr, "Assertion failed: %s\n  Message: %s\n  File: %s, Line: %d\n", \
                #cond, msg, __FILE__, __LINE__); \
        abort(); \
      } \
    } while(0)
#endif