#ifndef INCLUDE_LAYER_SIGMOID_LAYER_HPP
#define INCLUDE_LAYER_SIGMOID_LAYER_HPP
#include "layer/abstract/layer.hpp"
#include "ops/sigmoid_op.hpp"
namespace wbee_infer {

    class SigmoidLayer : public Layer {
        public:
            ~SigmoidLayer() override = default;

            explicit SigmoidLayer(const std::shared_ptr<Operator> &op);

            void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

            static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
        private:
            std::unique_ptr<SigmoidOperator> op_;    
    };
}
#endif
