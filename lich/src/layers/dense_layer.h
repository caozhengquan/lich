#ifndef LICH_SRC_LAYERS_DENSE_LAYER_H_
#define LICH_SRC_LAYERS_DENSE_LAYER_H_

#include "lich/src/layer.h"
#include "lich/src/tensor.h"

namespace lich {
  
template <typename Dtype>
class DenseLayer : public Layer<Dtype> {
 public:
  explicit DenseLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
   
 protected:
  virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top);

  virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
                       const vector<Tensor<Dtype>*>& top);

  virtual void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top);
                           
  virtual void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                           const vector<Tensor<Dtype>*>& bottom);   
  
  bool bias_term_;
  // [M_ * num_input_] * [num_input_ * num_output_] = [M_ * num_output] 
  int M_;
  int num_output_;
  int num_input_;
  Tensor<Dtype> bias_multi_;                    
};

} // namespace lich

#endif