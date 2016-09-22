#ifndef LICH_SRC_LAYERS_SOFTMAX_LAYER_H_
#define LICH_SRC_LAYERS_SOFTMAX_LAYER_H_

#include "lich/src/layer.h"
#include "lich/src/tensor.h"

namespace lich {
  
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
                       const vector<Tensor<Dtype>*>& top) override;
 protected:
  // For example, if the shape of bottom data is N * C * W * H and 
  // softmax_axis_ is 1. Then we do softmax for every point in W * H
  // with size C. In this example, outer_num_ is N, channel_ is C and
  // inner_num_ is W*H
  // Axis to do softmax
  int softmax_axis_;
  int outer_num_;
  int channels_;
  int inner_num_;
  // Helper data, scale_data used to scale the data to prevent overflow
  // while calculating exponents, given fact that:
  // exp(a) / exp(a) + exp(b) = exp(a - max) / exp(a - max) + exp(b - max)
  Tensor<Dtype> scale_data_;
  // multiplier is a matrix full of 1
  Tensor<Dtype> multiplier_;

  virtual void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) override;
                           
  virtual void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                           const vector<Tensor<Dtype>*>& bottom) override;   
                
};

} // namespace lich

#endif