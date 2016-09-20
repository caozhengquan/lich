#include "lich/src/layers/dense_layer.h"
#include "lich/src/filler.h"
#include "lich/lib/macros.h"
#include "lich/src/layer_factory.h"
#include "lich/lib/math.h"

#include <memory>

namespace lich {


template <typename Dtype>
void DenseLayer<Dtype>::LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
                                   const vector<Tensor<Dtype>*>& top) {
  const int num_output = this->layer_param_.dense_param().num_output();
  num_output_ = num_output;
  bias_term_ = this->layer_param_.dense_param().bias_term();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.dense_param().axis());
  num_input_ = bottom[0]->count(axis);
  if (bias_term_) {
      this->tensors_.resize(2);
  }                    
  else this->tensors_.resize(1);
  // Allocate weight space and fill weight with weight_filler. 
  this->tensors_[0].reset(new Tensor<Dtype>({num_input_, num_output_}));
  std::shared_ptr<Filler<Dtype>> weight_filler(Filler<Dtype>::GetFiller(
      this->layer_param_.dense_param().weight_filler()));
  weight_filler->Fill(this->tensors_[0].get());
  // If has bias, allocate bias space and fill bias with bias_filler.
  if (bias_term_) {
    this->tensors_[1].reset(new Tensor<Dtype>({1, num_output_}));
    std::shared_ptr<Filler<Dtype>> bias_filler(Filler<Dtype>::GetFiller(
        this->layer_param_.dense_param().bias_filler()));
    bias_filler->Fill(this->tensors_[1].get());
  }
}

template <typename Dtype>
void DenseLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                const vector<Tensor<Dtype>*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.dense_param().axis());
  const int num_input = bottom[0]->count(axis);
  CHECK_EQ(num_input, num_input_);
  M_ = bottom[0]->count(0, axis);
  // Reshape the top tensors
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = num_output_;
  top[0]->Reshape(top_shape);
  // Reshape the multiplier for bias if has bias term.
  if (bias_term_) {
    bias_multi_.Reshape({M_, 1});
    lich_set(M_, Dtype(1), bias_multi_.mutable_data());
  }
}

template <typename Dtype>
void DenseLayer<Dtype>::ForwardCpu(const std::vector<Tensor<Dtype>*>& bottom,
                                   const std::vector<Tensor<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* top_data = top[0]->mutable_data();
  
  // Compute X * W
  lich_gemm(CblasNoTrans, CblasNoTrans, M_, num_output_, num_input_,
            Dtype(1), bottom_data, this->tensors_[0]->data(),
            Dtype(0), top_data);
  // Compute 1 * bias
  if (bias_term_) {
    lich_gemm(CblasNoTrans, CblasNoTrans, M_, num_output_, 1,
              Dtype(1), bias_multi_.data(), this->tensors_[1]->data(),
              Dtype(1), top[0]->mutable_data());     
  }        
}

template <typename Dtype>
void DenseLayer<Dtype>::BackwardCpu(const vector<Tensor<Dtype>*>& top,
                                    const vector<Tensor<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->diff();
  Dtype* bottom_diff = bottom[0]->mutable_diff();
  const Dtype* bottom_data = bottom[0]->mutable_data();
  const Dtype* weight = this->tensors_[0]->data();
  Dtype* weight_diff = this->tensors_[0]->mutable_diff();
  // Update weight gradient: gradient += bottom * delta(l+1)
  lich_gemm(CblasTrans, CblasNoTrans, num_input_, num_output_, M_,
            Dtype(1), bottom_data, top_diff, Dtype(1), weight_diff);
  // Update bias gradient: bias += delta(l+1) if has bias
  if (bias_term_) {
    Dtype* bias_diff = this->tensors_[1]->mutable_diff();
    lich_gemm(CblasTrans, CblasNoTrans, 1, num_output_, M_,
              Dtype(1), bias_multi_.data(), top_diff, Dtype(1), bias_diff); 
  }  
  // Error propagating down: Delta(l) += delta(l+1) * W
  lich_gemm(CblasNoTrans, CblasTrans, M_, num_input_, num_output_,
            Dtype(1), top_diff, weight, Dtype(1), bottom_diff);
}

INSTANTIATE_CLASS(DenseLayer);
REGISTER_LAYER(Dense);

} // namespace lich 