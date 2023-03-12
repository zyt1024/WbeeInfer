#include "parser/parse_expression.hpp"
#include "ops/expression_op.hpp"
#include "layer/details/expression_layer.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"

static void ShowNodes(const std::shared_ptr<wbee_infer::TokenNode> &node) {
  if (!node) {
    return;
  }
  // 中序遍历的顺序
  ShowNodes(node->left);
  if (node->num_index < 0) {
    if (node->num_index == int(wbee_infer::TokenType::TokenAdd)) {
      LOG(INFO) << "ADD";
    } else if (node->num_index == int(wbee_infer::TokenType::TokenMul)) {
      LOG(INFO) << "MUL";
    }
  } else {
    LOG(INFO) << "NUM: " << node->num_index;
  }
  ShowNodes(node->right);
}

static void ShowNodes(const std::vector<std::shared_ptr<wbee_infer::TokenNode>> &nodes) {
  if (nodes.empty()) {
    return;
  }
  for(std::shared_ptr<wbee_infer::TokenNode> node: nodes){
    if (node->num_index < 0) {
        if (node->num_index == int(wbee_infer::TokenType::TokenAdd)) {
        LOG(INFO) << "ADD";
        } else if (node->num_index == int(wbee_infer::TokenType::TokenMul)) {
        LOG(INFO) << "MUL";
        }
    } else {
        LOG(INFO) << "NUM: " << node->num_index;
    }
  }
}

TEST(test_expression, expression1) {
  using namespace wbee_infer;
  const std::string &statement = "add(@1,@2)";
  ExpressionParser parser(statement);
  const auto &token_node  = parser.Generate_TokenNode();
  ShowNodes(token_node);
  const std::vector<std::shared_ptr<TokenNode>> nodes = parser.Generate();
  ShowNodes(nodes);
}

TEST(test_expression, expression2) {
  using namespace wbee_infer;
  const std::string &statement = "add(mul(@0,@1),@2)";
  ExpressionParser parser(statement);
  const auto &token_node  = parser.Generate_TokenNode();
  ShowNodes(token_node);
  const std::vector<std::shared_ptr<TokenNode>> nodes = parser.Generate();
  ShowNodes(nodes);
}

TEST(test_expression, expression3) {
  using namespace wbee_infer;
  const std::string &statement = "add(mul(@0,@1),mul(@2,add(@3,@4)))";
  ExpressionParser parser(statement);
  const auto &token_node  = parser.Generate_TokenNode();
  ShowNodes(token_node);
  const std::vector<std::shared_ptr<TokenNode>> nodes = parser.Generate();
  ShowNodes(nodes);
}

// TEST(test_expression, expression4) {
//   using namespace wbee_infer;
//   const std::string &statement = "add(div(@0,@1),@2)";
//   ExpressionParser parser(statement);
//   const auto &token_node  = parser.Generate_TokenNode();
//   ShowNodes(token_node);
//   const std::vector<std::shared_ptr<TokenNode>> nodes = parser.Generate();
//   ShowNodes(nodes);
// }


TEST(test_expression, add) {
  using namespace wbee_infer;
  const std::string &expr = "add(@0,@1)";
  std::shared_ptr<ExpressionOp> expression_op = std::make_shared<ExpressionOp>(expr);
  ExpressionLayer layer(expression_op);
  std::vector<std::shared_ptr<Tensor<float> >> inputs;
  std::vector<std::shared_ptr<Tensor<float> >> outputs;

  int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
    input->Fill(1.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
    input->Fill(2.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(3, 224, 224);
    outputs.push_back(output);
  }
  layer.Forwards(inputs, outputs);
  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 3.f);
    }
  }
}


// TEST(test_expression, complex) {
//   using namespace wbee_infer;
//   const std::string &expr = "add(mul(@0,@1),@2)";
//   std::shared_ptr<ExpressionOp> expression_op = std::make_shared<ExpressionOp>(expr);
//   ExpressionLayer layer(expression_op);
//   std::vector<std::shared_ptr<Tensor<float> >> inputs;
//   std::vector<std::shared_ptr<Tensor<float> >> outputs;

//   int batch_size = 4;
//   for (int i = 0; i < batch_size; ++i) {
//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
//     input->Fill(1.f);
//     inputs.push_back(input);
//   }

//   for (int i = 0; i < batch_size; ++i) {
//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
//     input->Fill(2.f);
//     inputs.push_back(input);
//   }

//   for (int i = 0; i < batch_size; ++i) {
//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
//     input->Fill(3.f);
//     inputs.push_back(input);
//   }

//   for (int i = 0; i < batch_size; ++i) {
//     std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(3, 224, 224);
//     outputs.push_back(output);
//   }
//   layer.Forwards(inputs, outputs);
//   for (int i = 0; i < batch_size; ++i) {
//     const auto &result = outputs.at(i);
//     for (int j = 0; j < result->size(); ++j) {
//       ASSERT_EQ(result->index(j), 5.f);
//     }
//   }
// }
