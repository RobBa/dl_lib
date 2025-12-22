/**
 * @file sequential.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "sequential.h"

using namespace std;
using namespace layers;

/**
 * @brief Returns true if dimensions valid, else false. 
 * Ensures consistency along network.
 */
bool SequentialNetwork::assertDims(const LayerBase& layer) const noexcept {
  if(layers.size() == 0)
    return true;

  return layers.at(layers.size()-1).getDims() == layer.getDims(); 
}