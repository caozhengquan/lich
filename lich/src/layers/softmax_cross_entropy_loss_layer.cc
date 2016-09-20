#include "lich/src/layers/softmax_cross_entropy_loss_layer.h"

#include <limits>

namespace lich {

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Tensor<Dtype>*>& bottom,
    const vector<Tensor<Dtype>*>& top) {
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(softmax_output_.get());
  softmax_layer_.SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Tensor<Dtype>*>& bottom,
    const vector<Tensor<Dtype>*>& top) {
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  dim_ = bottom[0]->count(softmax_axis_);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count()) <<
      "SoftmaxCrossEntropyLossLayer label size not correct!";
  softmax_layer_.Reshape(softmax_bottom_vec_, softmax_top_vec_);
  top[0]->Reshape(vector<int>()); // Scalar output is 0 axes.
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::ForwardCpu(
    const vector<Tensor<Dtype>*>& bottom,
    const vector<Tensor<Dtype>*>& top) {
  softmax_layer_.Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob = softmax_output_->data();
  const Dtype* label = bottom[1]->data();
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      CHECK_GE(label_value, 0);
      CHECK_LT(label_value, softmax_output_->shape(softmax_axis_));
      loss -= std::log(std::max(prob[i * dim_ + inner_num_ * label_value + j],
                                std::numeric_limits<float>::min()));
      ++count;
    }
  }
  top[0]->mutable_data()[0] = loss / count;
} 

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::BackwardCpu(
    const vector<Tensor<Dtype>*>& top,
    const vector<Tensor<Dtype>*>& bottom) {
  const Dtype* prob = softmax_output_->data();
  const Dtype* label = bottom[1]->data();
  Dtype* bottom_diff = bottom[0]->mutable_diff();
  // Compute error: if i == label, error = yi - 1;
  //                if i != label, error = yi
  lich_copy(softmax_output_->count(), prob, bottom_diff);
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
      ++count;
    }
  }
  // Scale by loss_weight
  const Dtype loss_weight = top[0]->diff()[0];
  lich_scal(bottom_diff_->count(), loss_weight / count, bottom_diff);
}

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossLayer);
REGISTER_LAYER(SoftmaxCrossEntropyLoss);

} // namespace lich 