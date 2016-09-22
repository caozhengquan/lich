#include "lich/src/layers/softmax_layer.h"
#include "lich/lib/math.h"
#include "lich/src/layer_factory.h"
#include "lich/lib/macros.h"

namespace lich {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                  const vector<Tensor<Dtype>*>& top) {
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  // 可以理解为channels_是类别个数，inner_num_是需要分类的data个数. 
  // 例如对于图像[batch, channel, height, width]，inner_num_是像素点个数height * width;
  // 对于文本[batch, class_num], inner_num_就是１.
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  channels_ = bottom[0]->shape(softmax_axis_);
  // Set up top tensors
  top[0]->Reshape(bottom[0]->shape());
  // Set up scale_data_.
  vector<int> scale_shape = bottom[0]->shape();
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
  Dtype* scale_data = scale_data_.mutable_data();
  lich_copy(bottom[0]->count(), bottom_data, top_data);
  const int dim = bottom[0]->count() / outer_num_;

  for (int i = 0; i < outer_num_; ++i) {
    // Calculate the max data for all channels 
    lich_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels_; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        scale_data[k] = std::max(scale_data[k], 
                                 bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // Substract the bottom data by max data
    lich_gemm(CblasNoTrans, CblasNoTrans, channels_, inner_num_, 1,
              Dtype(-1), multiplier_.data(), scale_data, Dtype(1), top_data);
    // Calculate exponets
    lich_exp(dim, top_data, top_data);
    // Sum exps
    lich_gemv(CblasTrans, channels_, inner_num_, Dtype(1), top_data,
              multiplier_.data(), Dtype(0), scale_data);
    // Divide by sum
    for (int j = 0; j < channels_; ++j) {
      lich_div(inner_num_, top_data, scale_data, top_data);
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
  Dtype* scale_data = scale_data_.mutable_data();
  lich_copy(top[0]->count(), top_diff, bottom_diff);
  const int dim = channels_ * inner_num_;
  // Compute error: delta_i = y_i * (delta_i - sigma_k(delta_k * y_k))
  for (int i = 0; i < outer_num_; ++i) {
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = lich_strided_dot(
        channels_, bottom_diff + i * dim + k, inner_num_,
        top_data + i * dim + k, inner_num_);
    }
    lich_gemm(CblasNoTrans, CblasNoTrans, channels_, inner_num_, 1,
              Dtype(-1), multiplier_.data(), scale_data, 
              Dtype(1), bottom_diff + i * dim);
  }
  lich_mul(top[0]->count(), top_data, bottom_diff, bottom_diff);
}

INSTANTIATE_CLASS(SoftmaxLayer);
REGISTER_LAYER(Softmax);

} // namespace lich