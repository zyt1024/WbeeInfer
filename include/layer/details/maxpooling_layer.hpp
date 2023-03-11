#ifndef INCLUDE_MAXPOOLING_LAYER_H
#define INCLUDE_MAXPOOLING_LAYER_H
#include "layer/abstract/layer.hpp"
#include "ops/maxpooling_op.hpp"
namespace wbee_infer {
    class MaxPoolingLayer : public Layer {
        public:
            // constructor
            explicit MaxPoolingLayer(const std::shared_ptr<Operator> &op);

            void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &input,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

            static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
        private:
            std::unique_ptr<MaxPoolingOp> op_;
    };
}
#endif