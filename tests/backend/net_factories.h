/**
 * @file net_factories.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "data_modeling/device.h"
#include "module/networks/sequential.h"
#include "module/layers/ff_layer.h"

#include "module/activation_functions/sigmoid.h"
#include "module/activation_functions/relu.h"
#include "module/activation_functions/leaky_relu.h"
#include "module/activation_functions/softmax.h"

#include <memory>

static std::shared_ptr<module::Sequential> makeBinaryNet(Device d = Device::CPU) {
  auto net = std::make_shared<module::Sequential>();

  net->append(std::make_shared<module::FfLayer>(2, 4, d, true, true));
  net->append(std::make_shared<module::LeakyReLu>(0.01));
  net->append(std::make_shared<module::FfLayer>(4, 1, d, true, true));
  net->append(std::make_shared<module::Sigmoid>());

  return net;
}

static std::shared_ptr<module::Sequential> makeBinaryNet2(Device d = Device::CPU) {
  auto net = std::make_shared<module::Sequential>();

  net->append(std::make_shared<module::FfLayer>(2, 4, d, true, true));
  net->append(std::make_shared<module::LeakyReLu>(0.01));
  net->append(std::make_shared<module::FfLayer>(4, 1, d, true, true));

  return net;
}

static std::shared_ptr<module::Sequential> makeMulticlassNet(Device d = Device::CPU) {
  auto net = std::make_shared<module::Sequential>();

  net->append(std::make_shared<module::FfLayer>(2, 8, d, true, true));
  net->append(std::make_shared<module::LeakyReLu>(0.01));
  net->append(std::make_shared<module::FfLayer>(8, 3, d, true, true));
  net->append(std::make_shared<module::Softmax>());

  return net;
}

static std::shared_ptr<module::Sequential> makeMulticlassNet2(Device d = Device::CPU) {
  auto net = std::make_shared<module::Sequential>();

  net->append(std::make_shared<module::FfLayer>(2, 8, d, true, true));
  net->append(std::make_shared<module::LeakyReLu>(0.01));
  net->append(std::make_shared<module::FfLayer>(8, 3, d, true, true));

  return net;
}
