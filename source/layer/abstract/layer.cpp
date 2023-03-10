#include "layer/abstract/layer.hpp"
#include <glog/logging.h>
namespace wbee_infer {
    void Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        LOG(FATAL) << "The layer " << this->layer_name_ << " not implement yet!";
    }
}