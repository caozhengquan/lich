#ifndef LICH_SRC_LAYERS_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_H_
#define LICH_SRC_LAYERS_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_H_

#include "lich/proto/layer_param.pb.h"
#include "lich/src/tensor.h"
#include "lich/src/layers/softmax_layer.h"

namespace lich {

class SoftmaxCrossEntropyLossLayer : public Layer<Dtype> {
 public:
  SoftmaxCrossEntropyLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        softmax_layer_(new SoftmaxLayer<Dtype>(param)),
        softmax_output_(new Tensor<Dtype>()) {}
 
 protected:
  shared_ptr<Tensor<Dtype>> softmax_output_;
  shared_ptr<SoftmaxLayer<Dtype>> softmax_layer_;
  vector<Tensor<Dtype>*> softmax_bottom_vec_;
  vector<Tensor<Dtype>*> softmax_top_vec_;
  int softmax_axis_, outer_num_, inner_num_, dim_;

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