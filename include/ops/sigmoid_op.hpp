#ifndef INCLUDE_OP_SIGMOID_OP_HPP
#define INCLUDE_OP_SIGMOID_OP_HPP
#include "op.hpp"

namespace wbee_infer {
    class SigmoidOperator : public Operator {
        public:
            explicit SigmoidOperator();
    };
}
#endif