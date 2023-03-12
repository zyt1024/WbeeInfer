//
// Created by fss on 22-12-18.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>

namespace wbee_infer {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
}

Tensor<float>::Tensor(const Tensor &tensor) {
  this->data_ = tensor.data_;
  this->raw_shapes_ = tensor.raw_shapes_;
}

Tensor<float> &Tensor<float>::operator=(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

void Tensor<float>::set_data(const arma::fcube &data) {
  CHECK(data.n_rows == this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const {
  return this->data_.empty();
}

//后面加上const表明只读
float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size());
  return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size());
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::fcube &Tensor<float>::data() {
  return this->data_;
}

const arma::fcube &Tensor<float>::data() const {
  return this->data_;
}

arma::fmat &Tensor<float>::at(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat &Tensor<float>::at(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  //todo 请把代码补充在这里1
  //1.申请一块空间(row,col,channel)
  arma::fcube new_data(this->data_.n_rows + pad_rows1 + pad_rows2,this->data_.n_cols + pad_cols1 + pad_cols2, this->data_.n_slices);
  new_data.fill(padding_value); //填充

  // 

  LOG(INFO) << "sss" << pad_rows1 << "  " << pad_cols1;
  // 查看子cube 将除了填充之外的赋值给原先的值
  new_data.subcube(pad_rows1,pad_cols1,0,new_data.n_rows - pad_rows2 - 1,new_data.n_cols - pad_cols2 - 1,
  new_data.n_slices - 1
  ) = this->data_;
  LOG(INFO) << new_data;
  this->data_ = std::move(new_data);
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float> &values) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t planes = rows * cols;
  const uint32_t channels = this->data_.n_slices;

  //todo 请把代码补充在这里2
  for(uint32_t i = 0; i < channels; ++i){
    auto& channel_data = this->data_.slice(i);
    const arma::fmat& channel_data_t = arma::fmat(values.data() + i*planes, this->cols(), this->rows());
    channel_data = channel_data_t.t();
  }
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor<float>::Flatten() {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  arma::fcube linear_cube(size, 1, 1);

  uint32_t channel = this->channels();
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();
  uint32_t index = 0;

  for (uint32_t c = 0; c < channel; ++c) {
    const arma::fmat &matrix = this->data_.slice(c);

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c_ = 0; c_ < cols; ++c_) {
        linear_cube.at(index, 0, 0) = matrix.at(r, c_);
        index += 1;
      }
    }
  }
  CHECK_EQ(index, size);
  this->data_ = linear_cube;
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->data_.fill(1.);
}

std::shared_ptr<Tensor<float>> Tensor<float>::Clone() {
  return std::make_shared<Tensor>(*this);
}

arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}


void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
  } else {
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
  }
}

void TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
  } else {
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
  }
}

std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    sftensor output_tensor = TensorCreate(tensor1->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    sftensor output_tensor = TensorCreate(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    return output_tensor;
  }
}

std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    sftensor output_tensor = TensorCreate(tensor1->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    sftensor output_tensor = TensorCreate(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    return output_tensor;
  }
}


std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes) {
  CHECK(shapes.size() == 3);
  return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}

std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& s1,
                                               const sftensor& s2) {
  CHECK(s1 != nullptr && s2 != nullptr);
  if (s1->shapes() == s2->shapes()) {
    return {s1, s2};
  } else {
    CHECK(s1->channels() == s2->channels());
    if (s2->rows() == 1 && s2->cols() == 1) {
      sftensor s2_ = TensorCreate(s2->channels(), s1->rows(), s1->cols());
      CHECK(s2->size() == s2->channels());
      for (uint32_t c = 0; c < s2->channels(); ++c) {
        s2_->slice(c).fill(s2->index(c));
      }
      return {s1, s2_};
    } else if (s1->rows() == 1 && s1->cols() == 1) {
      sftensor s1_ = TensorCreate(s1->channels(), s2->rows(), s2->cols());
      CHECK(s1->size() == s1->channels());
      for (uint32_t c = 0; c < s1->channels(); ++c) {
        s1_->slice(c).fill(s1->index(c));
      }
      return {s1_, s2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {s1, s2};
    }
  }
}


}


