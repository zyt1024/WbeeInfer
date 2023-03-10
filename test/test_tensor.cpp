#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>
#include "data/tensor.hpp"

TEST(test_tensor, create) {
  using namespace wbee_infer;
  Tensor<float> tensor(3, 32, 32);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 32);
  ASSERT_EQ(tensor.cols(), 32);
}

TEST(test_tensor, fill) {
  using namespace wbee_infer;
  Tensor<float> tensor(3, 3, 3);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 3);
  ASSERT_EQ(tensor.cols(), 3);

  std::vector<float> values;
  for (uint32_t i = 0; i < 27; ++i) {
    values.push_back((float) i);
  }
  tensor.Fill(values);
  LOG(INFO) << tensor.data();

  uint32_t index = 0;
  for (uint32_t c = 0; c < tensor.channels(); ++c) {
    for (uint32_t r = 0; r < tensor.rows(); ++r) {
      for (uint32_t c_ = 0; c_ < tensor.cols(); ++c_) {
        ASSERT_EQ(values.at(index), tensor.at(c, r, c_));
        index += 1;
      }
    }
  }
  LOG(INFO) << "Test fill passed!";
}

TEST(test_tensor, padding1) {
  using namespace wbee_infer;
  Tensor<float> tensor(3, 3, 3);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 3);
  ASSERT_EQ(tensor.cols(), 3);

  tensor.Fill(1.f); // 填充为1
  tensor.Padding({1, 1, 1, 1}, 0); // 边缘填充为0
  ASSERT_EQ(tensor.rows(), 5);
  ASSERT_EQ(tensor.cols(), 5);

  uint32_t index = 0;
  // 检查一下边缘被填充的行、列是否都是0
  for ( uint32_t c = 0; c < tensor.channels(); ++c) {
    for (uint32_t r = 0; r < tensor.rows(); ++r) {
      for (uint32_t c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ == 0 || r == 0) {
          ASSERT_EQ(tensor.at(c, r, c_), 0);
        }
        index += 1;
      }
    }
  }
  LOG(INFO) << "Test padding passed!";
}
