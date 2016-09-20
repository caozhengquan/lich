#include "lich/src/layers/sigmoid_cross_entropy_loss_layer.h"

#include <cmath>

namespace lich {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Tensor<Dtype>*>& bottom, 
    const vector<Tensor<Dtype>*>& top) {
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Tensor<Dtype>*>& bottom, 
    const vector<Tensor<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << 
      "SigmoidCrossEntropyLossLayer inputs must have the same count.";
  sigmoid_layer_.Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  top[0]->Reshape(vector<int>()); // Scalar output is 0 axes.
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::ForwardCpu(
    const vector<Tensor<Dtype>*>& bottom, 
    const vector<Tensor<Dtype>*>& top) {
  sigmoid_layer_.Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* input = bottom[0]->data();
  const Dtype* label = bottom[1]->data();
  Dtype* top_data = top[0]->mutable_data();
  const int count = bottom[0]->count();
  const int batch_num = bottom[0]->shape(0);
  Dtype loss = 0;
  // This formulation follows the definition in tensorflow:
  // https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#sigmoid_cross_entropy_with_logits
  for (int i = 0; i < count; ++i) {
    loss += std::max(input[i], Dtype(0)) - input[i] * label[i] 
        std::log(1 + std::exp(-(input[i] >= 0 ? input[i] : -input[i])));
  }
  top[0]->mutable_data()[0] = loss / batch_num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::BackwardCpu(
    const vector<Tensor<Dtype>*>& top,
    const vector<Tensor<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const int batch_num = bottom[0]->shape(0);
  const Dtype* sigmoid_output_data = sigmoid_output_->data();
  const Dtype* label = bottom[1]->data();
  Dtype* bottom_diff = bottom[0]->mutable_diff();
  // Calculate error(gradient)
  // grad(Loss, input) = sigmoid_out_put - label
  lich_sub(count, sigmoid_output_data, label, bottom_diff);
  // Scale by loss_weight
  // TODO(wzpfish): why caffe use loss_weight / batch_size to scale???
  const Dtype loss_weight = top[0]->diff()[0];
  lich_scal(count, loss_weight / batch_num, bottom_diff);
}

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER(SigmoidCrossEntropyLoss);

} // namespace lich