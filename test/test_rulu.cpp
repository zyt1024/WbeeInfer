#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.hpp"
#include "layer/details/relu_layer.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "data/tensor.hpp"
#include <armadillo>
using namespace wbee_infer;

TEST(test_layer, forward_relu1) {
  // using namespace wbee_infer;
  float thresh = 0.f;
  // 初始化一个relu operator 并设置属性
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);

  // 有三个值的一个tensor<float>  (c,h,w)
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  arma::fcube data1(1,3,1);
  data1.at(0) = -1.f;
  data1.at(1) = -2.f;
  data1.at(2) = 3.f;

  input->set_data(data1);
  // input->index(0) = -1.f; //output对应的应该是0
  // input->index(1) = -2.f; //output对应的应该是0
  // input->index(2) = 3.f; //output对应的应该是3
  // 主要第一个算子，经典又简单，我们这里开始！

  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理

  std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
  inputs.push_back(input);
  ReluLayer layer(relu_op);
  // 因为是4.1 所以没有作业 4.2才有
// 一个批次是1
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }

  LOG(INFO) << " test forward_relu1 success";
}

TEST(test_layer, forward_relu2) {
  
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
  std::shared_ptr<Layer> relu_layer = LayerRegisterer::CreateLayer(relu_op);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  relu_layer->Forwards(inputs, outputs);
  LOG(INFO) << " test " << input->index(0);
  ASSERT_EQ(outputs.size(), 1);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}