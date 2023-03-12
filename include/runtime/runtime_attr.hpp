#ifndef WBEE_INFER_RUNTIME_ATTR_HPP
#define WBEE_INFER_RUNTIME_ATTR_HPP
#include <vector>
#include <glog/logging.h>
#include "runtime_datatype.hpp"

namespace wbee_infer {
    struct RuntimeAttribute {
        std::vector<char> weight_data; // 节点中的权重参数
        std::vector<int> shape; // 节点中的形状信息
        RuntimeDataType type = RuntimeDataType::kTypeUnknown; // 节点中的数据类型

        /**
         * 从节点中加载权重参数
         * @tparam T 权重类型
         * @return 权重参数数组
        */
        template<class T>
        std::vector<T> get();
    };

    template<class T>
    std::vector<T> RuntimeAttribute::get() {
        // 检查节点属性中的权重类型
        CHECK(!weight_data.empty());
        CHECK(type != RuntimeDataType::kTypeUnknown);
        std::vector<T> weights;//存储权重
        switch (type) {
            case RuntimeDataType::kTypeFloat32 : {
                const bool is_float = std::is_same<T, float>::value;
                CHECK_EQ(is_float, true);
                const uint32_t float_size = sizeof(float); 
                CHECK_EQ(weight_data.size() % float_size, 0);
                for (uint64_t i = 0; i < weight_data.size() / float_size; ++i) {
                    float weight = *((float *)weight_data.data() + i);
                    weights.push_back(weight);
                }
            }
            default: {
                LOG(FATAL) << "Unknown weight data type";
            }
        }
        return weights;
    };
}
#endif