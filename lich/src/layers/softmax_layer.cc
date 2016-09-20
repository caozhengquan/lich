#include "lich/src/layers/softmax_layer.h"
#include "lich/lib/math.h"

namespace lich {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                  const vector<Tensor<Dtype>*>& top) {
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  channels_ = bottom[0]->shape(softmax_axis_);
  // Set up scale_data_.
  vector<int> scale_shape = bottom[0].shape();
  scale_shape[softmax_axis_] = 1;
  scale_data_.Reshape(scale_shape);
  // Set up multiplier_ 
  multiplier_.Reshape({1, channels_});
  Dtype* multiplier_data = multiplier_.mutable_data();
  lich_set(channels_, Dtype(1), multiplier_data);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                                     const vector<Tensor<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* top_data = top[0]->mutable_data();
  lich_copy(bottom[0]->count(), bottom_data, top_data);
  const int dim = channels_ * inner_num_;

  for (int i = 0; i < outer_num_; ++i) {
    // Calculate the max data for all channels 
    lich_copy(inner_num_, bottom_data + dim, scale_data_);
    for (int j = 0; j < channels_; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        scale_data_[k] = std::max(scale_data_[k], 
                                  bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // Substract the bottom data by max data
    lich_gemm(CblasNoTrans, CblasNoTrans, channels_, inner_num_, 1,
              Dtype(-1), multiplier_, scale_data_, Dtype(1), top_data);
    // Calculate exponets
    lich_exp(dim, top_data, top_data);
    // Sum exps
    lich_gemv(CblasTrans, channels_, inner_num_, Dtype(1), top_data,
              multiplier_, Dtype(0), scale_data_);
    // Divide by sum
    for (int j = 0; j < channels_; ++j) {
      lich_div(inner_num_, top_data, scale_data_, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::BackwardCpu(const vector<Tensor<Dtype>*>& top,
                                      const vector<Tensor<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->diff();
  const Dtype* top_data = top[0]->data();
  Dtype* bottom_diff = bottom[0]->mutable_diff();
  lich_copy(top[0]->count(), top_diff, bottom_diff);
  const int dim = channels_ * inner_num_;
  // Compute error: delta_i = y_i * (delta_i - sigma_k(delta_k * y_k))
  for (int i = 0; i < outer_num_; ++i) {
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_[k] = lich_strided_dot(
        channels_, bottom_diff + i * dim + k, inner_num_,
        top_data + i * dim + k, inner_num_);
    }
    lich_gemm(CblasNoTrans, CblasNoTrans, channels_, inner_num_, 1,
              Dtype(-1), multiplier_->data(), scale_data_, 
              Dtype(1), bottom_diff + i * dim);
  }
  lich_mul(top[0]->count(), top_data, bottom_diff, bottom_diff);
}

} // namespace lich