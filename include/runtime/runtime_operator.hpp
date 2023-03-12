#ifndef WBEE_INFER_RUNTIME_OPERATOR_HPP
#define WBEE_INFER_RUNTIME_OPERATOR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
// #include <memory>
#include "layer/abstract/layer_factory.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"
#include "runtime_attr.hpp"
namespace wbee_infer {
    struct RuntimeOperator {
        ~RuntimeOperator() {
            for (const auto &param : this->params){
                delete param.second;
            }
        }
        std::string name;//计算节点的名称
        std::string type;//计算节点的类型
        std::shared_ptr<Layer> layer; // 节点对应的计算Layer

        std::vector<std::string> output_names; // 节点的输出点名称
        std::shared_ptr<RuntimeOperand> output_operands; // 节点的输出操作数

        std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands; //节点的输入操作数

        std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq; // 节点的输入操作数，顺序排列
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators; // 输出接待你的名称和节点对应

        std::map<std::string, RuntimeParameter *> params;  /// 算子的参数信息
        std::map<std::string, std::shared_ptr<RuntimeAttribute> > attribute; /// 算子的属性信息，内含权重信息

    };
}

#endif