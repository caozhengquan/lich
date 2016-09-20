#ifndef LICH_SRC_LAYERS_SIGMOID_LAYER_H_
#define LICH_SRC_LAYERS_SIGMOID_LAYER_H_

#include "lich/proto/layer_param.pb.h"
#include "lich/src/tensor.h"

namespace lich {

template <typename Dtype>
class SigmoidLayer : public Layer<Dtype> {
 public:
  SigmoidLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
 
 protected:
  virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
                       const vector<Tensor<Dtype>*>& top) override;

  virtual void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) override;
                           
  virtual void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                           const vector<Tensor<Dtype>*>& bottom) override;
};

} // namespace lich 

#endif 