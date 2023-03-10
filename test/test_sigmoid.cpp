#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/op.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "layer/details/sigmoid_layer.hpp"

using namespace wbee_infer;
TEST(test_layer, forward_sigmoid){
    std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();

    // 注册算子
    std::shared_ptr<Layer> sigmoid_layer = LayerRegisterer::CreateLayer(sigmoid_op);

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1,1,3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    
    //执行sigmoid
    sigmoid_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1); //outputs.size即 batch_szie了
    for(uint32_t i = 0; i < outputs.size(); ++i){
        ASSERT_EQ(outputs.at(i)->index(0), 1 / (1 + std::exp(-input->index(0))) );
        ASSERT_EQ(outputs.at(i)->index(1), 1 / (1 + std::exp(-input->index(1))));
        ASSERT_EQ(outputs.at(i)->index(2), 1 / (1 + std::exp(-input->index(2))));
    }

    LOG(INFO) << "Test sigmoid operator PASSED!!!";
}