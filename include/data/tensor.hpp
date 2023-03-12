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

  /**
   * 返回张量第channel通道中的数据
   * @param channel 需要返回的通道
   * @return 返回的通道
   */
  arma::fmat& slice(uint32_t channel);

  /**
   * 返回张量第channel通道中的数据
   * @param channel 需要返回的通道
   * @return 返回的通道
   */
  const arma::fmat& slice(uint32_t channel) const;
  
 private:
  std::vector<uint32_t> raw_shapes_;
  arma::fcube data_;
};

    /**
   * 张量相加
   * @param tensor1 输入张量1
   * @param tensor2 输入张量2
   * @return 张量相加的结果
   */
  std::shared_ptr<Tensor<float>> TensorElementAdd(
      const std::shared_ptr<Tensor<float>>& tensor1,
      const std::shared_ptr<Tensor<float>>& tensor2);

  /**
   * 张量相加
   * @param tensor1 输入张量1
   * @param tensor2 输入张量2
   * @param output_tensor 输出张量
   */
  void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                        const std::shared_ptr<Tensor<float>>& tensor2,
                        const std::shared_ptr<Tensor<float>>& output_tensor);

  /**
   * 矩阵点乘
   * @param tensor1 输入张量1
   * @param tensor2 输入张量2
   * @param output_tensor 输出张量
   */
  void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                            const std::shared_ptr<Tensor<float>>& tensor2,
                            const std::shared_ptr<Tensor<float>>& output_tensor);

  /**
   * 张量相乘
   * @param tensor1 输入张量1
   * @param tensor2 输入张量2
   * @return 张量相乘的结果
   */
  std::shared_ptr<Tensor<float>> TensorElementMultiply(
      const std::shared_ptr<Tensor<float>>& tensor1,
      const std::shared_ptr<Tensor<float>>& tensor2);

  /**
   * 创建一个张量
   * @param channels 通道数量
   * @param rows 行数
   * @param cols 列数
   * @return 创建后的张量
   */
  std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                              uint32_t cols);

  /**
   * 创建一个张量
   * @param shapes 张量的形状
   * @return 创建后的张量
   */
  std::shared_ptr<Tensor<float>> TensorCreate(
      const std::vector<uint32_t>& shapes);


  using ftensor = Tensor<float>;
  using sftensor = std::shared_ptr<Tensor<float>>;
  // 广播
  std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor &s1, const sftensor &s2);

}
#endif //INCLUDE_TENSOR_HPP_
