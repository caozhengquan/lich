#ifndef LICH_SRC_LAYERS_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_H_
#define LICH_SRC_LAYERS_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_H_

#include "lich/proto/layer_param.pb.h"
#include "lich/src/tensor.h"
#include "lich/src/layers/sigmoid_layer.h"

namespace lich {

class SigmoidCrossEntropyLossLayer : public Layer<Dtype> {
 public:
  SigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
        sigmoid_output_(new Tensor<Dtype>()) {}

 protected:
  shared_ptr<SigmoidLayer<Dtype>> sigmoid_layer_;
  shared_ptr<Tensor<Dtype>> sigmoid_output_;
  vector<Tensor<Dtype>*> sigmoid_bottom_vec_;
  vector<Tensor<Dtype>*> sigmoid_top_vec_;

  virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) override;

  virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
                       const vector<Tensor<Dtype>*>& top) override;

  virtual void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) override;
                           
  virtual void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                           const vector<Tensor<Dtype>*>& bottom) override;
};

} // namespace lich

#endif