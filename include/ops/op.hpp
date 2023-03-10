//
// Created by zyt on 22-12-20.
//
// 算子类
// 

#ifndef INCLUDE_OPS_OP_HPP_
#define INCLUDE_OPS_OP_HPP_
namespace wbee_infer {
  enum class OpType { // 算子类型枚举
    kOperatorUnknown = -1,
    kOperatorRelu = 0,
    kOperatorSigmoid = 1,
  };

  class Operator {
  public:
    OpType op_type_ = OpType::kOperatorUnknown; //不是一个具体节点 制定为unknown

    virtual ~Operator() = default; //

    explicit Operator(OpType op_type);
  };


}
#endif //INCLUDE_OPS_OP_HPP_
