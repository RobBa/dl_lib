#pragma once

#include "layer_base.h"
#include "initializers.h"

#include <vector>
#include <memory>

namespace layers {
    class ff_layer_cpu : public layer_base {
        protected:
            // the matrix we multiply with. For efficiency dim = (output_size, input_size)
            std::vector< std::vector<ftype> > weights;
            
            // memoization
            mutable ftype* v1;

            void reset_vector(ftype* v, int size) const noexcept;

        public:
            ff_layer_cpu(int in_size, int out_size);
            ~ff_layer_cpu() noexcept;

            ftype* forward(ftype* input) const override;
            //ftype* backward(ftype* input) override;
    };
}
