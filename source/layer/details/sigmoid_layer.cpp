#include "layer/details/sigmoid_layer.hpp"
#include <glog/logging.h>
#include "ops/sigmoid_op.hpp"
#include "layer/abstract/layer_factory.hpp" // 工厂hpp
#include <armadillo>

namespace wbee_infer {
    SigmoidLayer::SigmoidLayer(const std::shared_ptr<Operator> &op) : Layer("Sigmoid"){
        CHECK(op->op_type_ == OpType::kOperatorSigmoid) << "Operator has a wrong type: " << int(op->op_type_);

        /*
            dynamic_cast 判断一下op指针是不是指向一个SigmoidOperator类的指针
        */
       SigmoidOperator *sigmoid_op = dynamic_cast<SigmoidOperator *>(op.get());
       
        CHECK(sigmoid_op != nullptr) << "Sigmoid operator is empty";
        this->op_ = std::make_unique<SigmoidOperator>();
    }
    void SigmoidLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<float>>> &outputs){
                CHECK(this->op_ != nullptr);
                CHECK(this->op_->op_type_ == OpType::kOperatorSigmoid);
                CHECK(!inputs.empty());

                const uint32_t batch_size = inputs.size();
                for(uint32_t i = 0; i < batch_size; ++i){
                    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
                    std::shared_ptr<Tensor<float>> output_data = input_data->Clone();
                    // y=1/(1+e^{-x})
                    // 对张量中每一个元素进行运算，进行sigmoid
                    output_data->data().transform([](float value){
                        return 1.f / (1 + expf(-value));
                    });
                    outputs.push_back(output_data);
                }                            
    }
    std::shared_ptr<Layer> SigmoidLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
    CHECK(op != nullptr);
    CHECK(op->op_type_ == OpType::kOperatorSigmoid);

    std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
    return sigmoid_layer;
    }

    //创建一个KsigmoidLayer对象 向注册器中注册算子
    LayerRegistererWrapper kSigmoidLayer(OpType::kOperatorSigmoid, SigmoidLayer::CreateInstance);
}