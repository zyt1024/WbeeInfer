#ifndef INCLUDE_LAYER_LAYER_HPP_
#define INCLUDE_LAYER_LAYER_HPP_
#include <iostream>
#include <vector>
#include "data/tensor.hpp"
namespace wbee_infer{

    class Layer {
        public:
            // 单参数 构造函数,使用explicit防止 类型隐式转换
            explicit Layer(std::string layer_name): layer_name_(std::move(layer_name)) {};

            // 每个算子的前向过程
            virtual void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                    std::vector<std::shared_ptr<Tensor<float>>> &outputs);

            // 析构函数
            virtual ~Layer() = default;
        private:
            std::string layer_name_; // layer的名称
    };
}
#endif