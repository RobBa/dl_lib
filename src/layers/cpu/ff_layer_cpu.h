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

#include <vector>
#include <memory>

namespace layers {
    class FfLayerCpu : public LayerBase {
        protected:
            // the matrix we multiply with. For efficiency dim = (output_size, input_size)
            std::vector< std::vector<ftype> > weights; // TODO: optimize via flattening out
            
            // memoization
            mutable ftype* v1;

            void resetVector(ftype* v, int size) const noexcept;

        public:
            FfLayerCpu(int in_size, int out_size);
            ~FfLayerCpu() noexcept;

            ftype* forward(ftype* input) const override;
            //ftype* backward(ftype* input) override;

            int n_rows() const noexcept { return 0; }
            int n_cols() const noexcept { return 0; }
    };
}
