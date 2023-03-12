#ifndef WBEE_INFER_RUNTIME_PARAMETER_HPP
#define WBEE_INFER_RUNTIME_PARAMETER_HPP

#include "status_code.hpp"
#include <vector>
#include <string>

namespace wbee_infer {
    /**
     * 计算节点中的参数信息，参数一共可以分为如下的几类
     * 1.int
     * 2.float
     * 3.string
     * 4.bool
     * 5.int array
     * 6.string array
     * 7.float array
     */
    struct RuntimeParameter { // 计算节点中的参数信息 基类
        virtual ~RuntimeParameter() = default;

        // constructor
        explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown) : type(type){

        } 
        RuntimeParameterType type = RuntimeParameterType::kParameterUnknown; // default is kParameterUnknown
    };

    struct RuntimeParameterInt : public RuntimeParameter {
        RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt){

        }
        int value = 0;
    };

    struct RuntimeParameterFloat : public RuntimeParameter {
    RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {

    }
    float value = 0.f;
    };

    struct RuntimeParameterString : public RuntimeParameter {
    RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {

    }
    std::string value;
    };

    struct RuntimeParameterIntArray : public RuntimeParameter {
    RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {

    }
    std::vector<int> value;
    };

    struct RuntimeParameterFloatArray : public RuntimeParameter {
    RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {

    }
    std::vector<float> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter {
    RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {

    }
    std::vector<std::string> value;
    };

    struct RuntimeParameterBool : public RuntimeParameter {
    RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {

    }
    bool value = false;
    };    
}
#endif