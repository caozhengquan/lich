#ifndef LICH_SRC_LAYERS_RELU_LAYER_H_
#define LICH_SRC_LAYERS_RELU_LAYER_H_

namespace lich {

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {
 public:
  ReluLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
 
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