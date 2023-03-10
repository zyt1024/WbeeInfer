//
// Created by fss on 22-12-18.
//

#ifndef INCLUDE_TENSOR_HPP_
#define INCLUDE_TENSOR_HPP_

#include <memory>
#include <vector>
#include <armadillo>

namespace wbee_infer {

template<typename T>
class Tensor {

};

template<>
class Tensor<uint8_t> {
  // 待实现
};

template<>
class Tensor<float> {
 public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  Tensor(const Tensor &tensor);

  Tensor<float> &operator=(const Tensor &tensor);

  uint32_t rows() const;

  uint32_t cols() const;

  uint32_t channels() const;

  uint32_t size() const;

  void set_data(const arma::fcube &data);

  bool empty() const;

  float index(uint32_t offset) const;

  // 这两个的区别
  float& index(uint32_t offset);
  
  std::vector<uint32_t> shapes() const;

  arma::fcube &data();

  const arma::fcube &data() const;

  arma::fmat &at(uint32_t channel);

  const arma::fmat &at(uint32_t channel) const;

  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  float &at(uint32_t channel, uint32_t row, uint32_t col);

  void Padding(const std::vector<uint32_t> &pads, float padding_value);

  void Fill(float value);

  void Fill(const std::vector<float> &values);

  void Ones();

  void Rand();

  void Show();

  void Flatten();
  std::shared_ptr<Tensor<float>> Clone();

 private:
  std::vector<uint32_t> raw_shapes_;
  arma::fcube data_;
};
}
#endif //INCLUDE_TENSOR_HPP_
