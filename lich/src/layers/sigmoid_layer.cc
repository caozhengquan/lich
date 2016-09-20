#include "lich/src/layers/sigmoid_layer.h"

#include <cmath>

namespace lich {

namespace {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1.0 / (1 + std::exp(-x));
}

}

template <typename Dtype>
void SigmoidLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                  const vector<Tensor<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                                     const vector<Tensor<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* top_data = top[0]->mutable_data();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::BackwardCpu(const vector<Tensor<Dtype>*>& top,
                                      const vector<Tensor<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->diff();
  const Dtype* top_data = top[0]->data();
  Dtype* bottom_diff = bottom[0]->mutable_diff();
  for (int i = 0; i < count; ++i) {
    bottom_diff[i] += top_diff[i] * top_data[i] * (1 - top_data[i]);
  }
}

INSTANTIATE_CLASS(SigmoidLayer);
REGISTER_LAYER(Sigmoid);

} // namespace lich 