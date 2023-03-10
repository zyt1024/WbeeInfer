//
// Created by zyt on 22-12-20.
//

#ifndef INCLUDE_OPS_RELU_OP_HPP_
#define INCLUDE_OPS_RELU_OP_HPP_

#include "op.hpp"

namespace wbee_infer {
class ReluOperator : public Operator {
 public:
  ~ReluOperator() override = default;

  explicit ReluOperator(float thresh);

  // 设置 thresh 阈值
  void set_thresh(float thresh);

  // 获得 thresh 阈值
  float get_thresh() const;

 private:
  // 需要传递到reluLayer中，怎么传递？
  float thresh_ = 0.f; // 用于过滤tensor<float>值当中大于thresh的部分
  // relu存的变量只有thresh
  // stride padding kernel_size 这些是到时候convOperator需要的
  // operator起到了属性存储、变量的作用
  // operator所有子类不负责具体运算
  // 具体运算由另外一个类Layer类负责
  // y =x  , if x >=0 y = 0 if x < 0

};
}
#endif // INCLUDE_OPS_RELU_OP_HPP_
