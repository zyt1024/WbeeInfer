#ifndef WBEE_INFER_EXPRESSION_OP_HPP
#define WBEE_INFER_EXPRESSION_OP_HPP
#include <vector>
#include <string>
#include <memory>
#include "ops/op.hpp"
#include "parser/parse_expression.hpp"

namespace wbee_infer {
    class ExpressionOp : public Operator {
        public:
            explicit ExpressionOp(const std::string &expr); 
            std::vector<std::shared_ptr<TokenNode>> Generate();

        private:
            std::shared_ptr<ExpressionParser> parser_;
            std::vector<std::shared_ptr<TokenNode>> nodes_; // nodes_ 表示经过逆波兰式之后得到得到节点
            std::string expr_; // 表达式字符串

    };
}
#endif