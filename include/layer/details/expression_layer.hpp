#ifndef WBEE_INFER_EXPRESSION_HPP
#define WBEE_INFER_EXPRESSION_HPP

#include "layer/abstract/layer.hpp"
#include "ops/expression_op.hpp"

namespace wbee_infer {
    class ExpressionLayer : public Layer {
        public:
            explicit ExpressionLayer(const std::shared_ptr<Operator> &op);
            
            void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        
        private:
            std::unique_ptr<ExpressionOp> op_;
    };
}

#endif