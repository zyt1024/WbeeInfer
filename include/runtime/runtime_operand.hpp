#ifndef WBEE_INFET_RUNTIME_OPERATOR_HPP
#define WBEE_INFET_RUNTIME_OPERATOR_HPP

#include <string>
#include <vector>
// #include "status_code.hpp"
#include "runtime_datatype.hpp"
#include <data/tensor.hpp>

namespace wbee_infer{
    struct RuntimeOperand {
        std::string name; // 操作数名称
        std::vector<int32_t> shapes; // 操作数的形状
        std::vector<std::shared_ptr<Tensor<float>>> datas; // 存储操作数
        RuntimeDataType type = RuntimeDataType::kTypeUnknown; // 操作数的类型，一般是float
    };
}
#endif