/**
 * @file ff_layer.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "layer_base.h"
#include "utility/initializers.h"

#include <optional>

namespace layers {
    class FfLayer : public LayerBase {
        public:
            FfLayer(const std::vector<tensorDim_t>& dims, bool useBias=true, bool requiresGrad=false);
            FfLayer(const std::vector<tensorDim_t>& dims, Device d, bool useBias=true, bool requiresGrad=false);

            Tensor forward(const Tensor& input) const override;
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) const override;

            void print(std::ostream& os) const noexcept override;
    };
}
