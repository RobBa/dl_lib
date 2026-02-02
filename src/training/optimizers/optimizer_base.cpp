/**
 * @file optimizer_base.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-02-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "optimizer_base.h"

float OptimizerBase::getLr() const noexcept {
    return lr;
}

void OptimizerBase::setLr(const float lr) noexcept {
    this->lr = lr;
}