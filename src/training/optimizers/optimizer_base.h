/**
 * @file optimizer_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "tensor.h"

class OptimizerBase {
    private:
        float lr = 0.05;

    public:
        virtual Tensor operator()(Tensor& t) const noexcept;
        float getLr() const noexcept;
        void setLr(const float lr) noexcept;
};