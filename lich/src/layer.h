#ifndef LICH_SRC_LAYER_H_
#define LICH_SRC_LAYER_H_

#include "lich/lib/macros.h"
#include "lich/src/common.h"
#include "lich/proto/layer_param.pb.h"
#include "lich/lib/math.h"
#include "lich/src/tensor.h"

#include <vector>
#include "glog/logging.h"

namespace lich {

template <typename Dtype>
class Layer {
 public:
  explicit Layer(const LayerParameter& param) : layer_param_(param) {}
  virtual ~Layer() {}
  
  void SetUp(const vector<Tensor<Dtype>*>& bottom,
             const vector<Tensor<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }
  // Return the loss while forward, loss only valid for loss layer.
  Dtype Forward(const vector<Tensor<Dtype>*>& bottom,
                const vector<Tensor<Dtype>*>& top);
  
  void BackWard(const vector<Tensor<Dtype>*>& top,
                const vector<Tensor<Dtype>*>& bottom);
  
  // Get and Set function for loss_weight_
  Dtype loss_weight(const int top_id) const {
    CHECK_GE(top_id, 0);
    return (loss_weight_.size() > top_id) ? loss_weight_[top_id] : Dtype(0);
  }
  
  void set_loss_weight(const int top_id, Dtype value) {
    CHECK_GE(top_id, 0);
    if (loss_weight_.size() <= top_id) {
      loss_weight_.resize(top_id + 1);
    }
    loss_weight_[top_id] = value;
  }
  
  vector<shared_ptr<Tensor<Dtype>>>& tensors() { return tensors_; }
  
 protected:
  LayerParameter layer_param_;
  Phase phase_;
  // The parameters for this layer.
  vector<shared_ptr<Tensor<Dtype>>> tensors_;
  // Flags to see which top tensor need to calculate loss in loss function.
  vector<Dtype> loss_weight_;
  
  // Typically allocate and initialize the learnable parameters for the layer if any.
  // May be override for different layers.
  virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) {}
  
  // Reshape the top tensors in appropriate shape.
  // May be override for different layers.
  virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
                       const vector<Tensor<Dtype>*>& top) {}   
  
  // Check whether the bottom and top is valid for the layer.
  // May be override for different layers. 
  virtual void CheckBlobCounts(const vector<Tensor<Dtype>*>& bottom,
                               const vector<Tensor<Dtype>*>& top) {}

  virtual void SetLossWeights(const vector<Tensor<Dtype>*>& top);

  virtual void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) = 0;
                          
  virtual void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                           const vector<Tensor<Dtype>*>& bottom) = 0;                        
  
 private:
  DISALLOW_COPY_AND_ASSIGN(Layer);
};

template <typename Dtype>
void Layer<Dtype>::SetLossWeights(const vector<Tensor<Dtype>*>& top) {
  const int num_loss_weights = layer_param_.loss_weight_size();
  if (num_loss_weights > 0) {
    CHECK_EQ(num_loss_weights, top.size()) << "Must specify one weight per top tensor";
    for (int i = 0; i < num_loss_weights; ++i) {
      const Dtype loss_weight = layer_param_.loss_weight(i);
      if (loss_weight == Dtype(0)) continue;
      set_loss_weight(i, loss_weight);
      const int count = top[i]->count();
      lich_set(count, loss_weight, top[i]->mutable_diff());
    }
  }
}

template <typename Dtype>
Dtype Layer<Dtype>::Forward(const vector<Tensor<Dtype>*>& bottom,
                            const vector<Tensor<Dtype>*>& top) {
  ForwardCpu(bottom, top);
  Dtype loss = 0;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    if (!loss_weight(top_id)) continue; 
    const int count = top[top_id]->count();
    const Dtype* data = top[top_id]->data();
    const Dtype* weight = top[top_id]->diff();
    loss += lich_dot(count, data, weight);
  }        
  return loss;
}

template <typename Dtype>
void Layer<Dtype>::BackWard(const vector<Tensor<Dtype>*>& top,
                            const vector<Tensor<Dtype>*>& bottom) {
  BackwardCpu(top, bottom);                                
}

} // namespace lich 

#endif