#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace wbee_infer {
    void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
        CHECK(creator != nullptr) << "Layer creator is empty";
        CreateRegistry &registry = Registry();
        CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type) << " has already registered!";
        registry.insert({op_type, creator});
    }


    std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op) {\

        CreateRegistry &registry = Registry(); // 如果已经存在则就不再创建
        
        const OpType op_type = op->op_type_;

        LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);

        // 从注册器中找到相应算子的初始化方法
        const auto &creator = registry.find(op_type)->second;

        LOG_IF(FATAL, !creator) << "Layer creator is empty!";
        
        // typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op)
        std::shared_ptr<Layer> layer = creator(op);// 

    
        LOG_IF(FATAL, !layer) << "Layer init failed!";
        return layer;
    }

    LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {

        // typedef std::map<OpType, Creator> CreateRegistry,创建一个map对象,这个map对象全局只有一份
        
        static CreateRegistry *kRegistry = new CreateRegistry();
        CHECK(kRegistry != nullptr) << "Global layer register init failed!";
        
        return *kRegistry;
    }
    // 调用链：LayerRegisterer::RegisterCreator -> Registry() , LayerRegisterer::CreateLayer ->Registry(),目的都是为了拿到一个map对象
}