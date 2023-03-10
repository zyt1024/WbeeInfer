//
// Created by zyt on 22-12-20.
// 创建一个工厂类,产生算子

#ifndef INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define INCLUDE_FACTORY_LAYER_FACTORY_HPP_

#include "ops/op.hpp"
#include "layer.hpp"
namespace wbee_infer {
    class LayerRegisterer{
        public: 

            // 定义函数指针类型
            /*
                定义了一个函数值指针变量Creator,它指向返回值为std::shared_ptr<Layer>, 
                参数为std::shared_ptr<Operator> &op的一类函数，
            */
            typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);

            // key是算子名, value 是 创建该层对应方法(Creator)
            typedef std::map<OpType, Creator> CreateRegistry;

            // 静态生命周期不在函数栈中
            static void RegisterCreator(OpType op_type, const Creator &creator);

            // 创建算子
            static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);

            static CreateRegistry &Registry(); // 创建注册表，全局只有一个


    };

    // 调用链条：relulayer -> LayerRegistererWrapper --> LayerRegisterer::RegisterCreator
    class LayerRegistererWrapper {
        public:
            //构造方法
            LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator) {
                LayerRegisterer::RegisterCreator(op_type, creator);
            }
    };
}
  
#endif
