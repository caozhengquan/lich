#include "lich/src/layers/relu_layer.h"

namespace lich {

template <typename Dtype>
void Relu<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void Relu<Dtype>::ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                             const vector<Tensor<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* top_data = top[0]->mutable_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(Dtype(0), bottom_data[i]);
  }
}

template <typename Dtype>
void Relu<Dtype>::BackwardCpu(const vector<Tensor<Dtype>*>& bottom,
                              const vector<Tensor<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* bottom_diff = bottom[0]->mutable_diff();
  const Dtype* top_diff = top[0]->diff();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] > 0) {
      bottom_diff[i] += top_diff[i];
    }
  }
}


} // namespace lich