/**
 * @file ff_layer_cpu.h
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
#include "initializers.h"
#include "tensor.h"

#include <vector>
#include <memory>

namespace layers {
    class FfLayerCpu : public LayerBase {
        protected:
            Dimension dims;

            Tensor weights;
            
            // memoization
            mutable ftype* v1;

            void resetVector(ftype* v, std::uint16_t size) const noexcept;

        public:
            FfLayerCpu(std::uint16_t in_size, std::uint16_t out_size);
            ~FfLayerCpu() noexcept;

            ftype* forward(ftype* input) const override;
            //ftype* backward(ftype* input) override;

            const Dimension& getDim() const noexcept { return dims; }
    };
}
