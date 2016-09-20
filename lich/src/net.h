#ifndef LICH_SRC_NET_H_
#define LICH_SRC_NET_H_

#include "lich/src/tensor.h"
#include "lich/src/common.h"

namespace lich {

template <typename Dtype>
class Net {
 public:
  Net() {}
  Net(const NetParameter& net_param);

  void Init(const NetParameter& net_param);
  
  void ClearDiffError();

  void AddLayer(const LayerParameter& param);

  Dtype ForwardBackWard();
  // Update the learnable parameter.
  void Update();

  const vector<shared_ptr<Tensor<Dtype>>>& learnable_params() const { return learnable_params_; }
  const vector<float>& params_weight_decay() const { return params_weight_decay_; }
  const vector<float>& params_lr() const { return params_lr_; }

 protected:
  vector<shared_ptr<Layer<Dtype>>> layers_;
  std::unordered_map<string, std::size_t> name_to_layer_index_;
  
  vector<shared_ptr<Tensor<Dtype>>> tensors_;
  std::unordered_map<string, std::size_t> name_to_tensor_index_;
  vector<vector<Tensor<Dtype>*>> bottom_tensors_;
  vector<vector<Tensor<Dtype>*>> top_tensors_;
  
  vector<Tensor<Dtype>*> learnable_params_;
  // vector<string> param_display_names_;
  std::unordered_map<string, std::size_t> name_to_learnable_index_;

  vector<float> params_weight_decay_;
  vector<float> params_lr_;
  
  // Check whether the bottom tensors are valid and restore them.
  void AddBottom(const LayerParameter& param);
  void AddTop(const LayerParameter& param);
  void AddTop(const LayerParameter& param);
 private:
  DISALLOW_COPY_AND_ASSIGN(Net);
};

} // namespace lich

#endif