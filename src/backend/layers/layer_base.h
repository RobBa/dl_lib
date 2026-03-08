/**
 * @file layer_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "data_modeling/tensor.h"
#include "utility/global_params.h"
#include "activation_functions/activation_function_base.h"

#include <optional>
#include <memory>

#include <iostream>

namespace layers {
    /** 
     * The base class for all the layers that we have. Not instantiable.
     */
    class LayerBase {       
        protected:
            bool requiresGrad = false;
            bool useBias = false;

            std::shared_ptr<Tensor> weights = nullptr;
            std::shared_ptr<Tensor> bias = nullptr;

            std::vector< std::shared_ptr<activation::ActivationFunctionBase> > activations;

        public:
            LayerBase(bool useBias, bool requiresGrad) 
                : useBias{useBias}, requiresGrad{requiresGrad}
            { }

            virtual ~LayerBase() noexcept = default;

            // for inference -> no graph creation
            virtual Tensor forward(const Tensor& input) const = 0;

            // for training -> creates graph
            virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) const = 0;

            // weights should always exist, never nullopt outside of c'tor
            const Dimension& getDims() const noexcept {
                assert(weights);
                return weights->getDims();
            }

            void addActivation(std::shared_ptr<activation::ActivationFunctionBase> f);

            auto getWeights() const noexcept { return weights; }
            auto getBias() const noexcept { return bias; }

            virtual void print(std::ostream& os) const noexcept;
            friend std::ostream& operator<<(std::ostream& os, const LayerBase& t) noexcept;
    };
}