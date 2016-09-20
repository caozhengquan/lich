#include "lich/src/net.h"
#include "lich/src/layer_factory.h"

#include "glog/logging.h"

namespace lich {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& net_param) {
  Init(net_param);  
}

template <typename Dtype>
void Net<Dtype>::init(const NetParameter& net_param) {
  // TODO(wzpfish): complete the init using net_param
  return;
}

template <typename Dtype>
void Net<Dtype>::ClearDiffError() {
  // Clear the gradients for all learnable parameters.
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Dtype* diff = learnable_params_[i]->mutable_diff();
    const int N = learnable_params_[i]->count();
    lich_set(N, Dtype(0), diff);
  }
  // Clear the errors for all layers
  for (int i = 0; i < top_tensors_.size(); ++i) {
    Dtype* error = top_tensors_[i]->mutable_diff();
    const int N = top_tensors_[i]->count();
    lich_set(N, Dtype(0), error);
  }
}

template <typename Dtype>
void Net<Dtype>::AddLayer(const LayerParameter& param) {
  const string& name = param.name();  
  CHECK_EQ(name_to_layer_index_.count(name), 0) << "Layer: " << name 
      << "exists.";
  shared_ptr<Layer<Dtype>> layer(LayerRegistry::Global()->CreateLayer(param));
  name_to_layer_index_[name] = layers_.size();
  layers_.push_back(layer);
  AddBottom(param);
  AddTop(param);
  layer->SetUp(bottom_tensors_.back(), top_tensors_.back());
  AddParam(param);
}

template <typename Dtype>
void Net<Dtype>::AddBottom(const LayerParameter& param) {
  vector<Tensor<Dtype>*> bottom_tensors(param.bottom_size());
  for (int i = 0; i < param.bottom_size(); ++i) {
    const string& name = param.bottom(i);
    CHECK_GT(name_to_tensor_index_[name], 0) << "Unknown bottom tensor: " << name;
    bottom_tensors[i] = tensors_[name_to_tensor_index_[name]].get();
  }
  bottom_tensors_.push_back(bottom_tensors);
}

template <typename Dtype>
void Net<Dtype>::AddTop(const LayerParameter& param) {
  vector<Tensor<Dtype>*> top_tensors(param.top_size());
  for (int i = 0; i < param.top_size(); ++i) {
    const string& name = param.top(i);
    // Check if in-place computation.
    if (param.bottom_size() > i && name == param.bottom(i)) {
      LOG(INFO) << "Inplace computation.";
    }
    else {
      CHECK_EQ(name_to_tensor_index_[name], 0) << "Duplicate top tensor: " << name;
      name_to_tensor_index_[name] = tensors_.size();
      tensors_.push_back(shared_ptr<Dtype>(new Tensor<Dtype>()));
    }
    top_tensors[i] = tensors_[name_to_tensor_index_[name]].get();
  }
  top_tensors_.push_back(top_tensors);
}

template <typename Dtype>
void Net<Dtype>::AddParam(const LayerParameter& param) {
  const int num_params = layers_.back()->tensors_.size();
  const int num_spec_params = param.param_size();
  const int layer_index = layers_.size() - 1;
  CHECK_LE(num_spec_params, num_params) << "Too much parameters specified to layer"
      << param.name() << ", need" << num_params << " parameters".
  for (int param_idx = 0; param_idx < num_params; ++param_idx) {
    string param_name = (param_idx < num_spec_params ? 
                         param.param(param_idx).name() : "");
    float decay_mult = (param_idx < num_spec_params ? 
                        param.param(param_idx).decay_mult() : 1);
    float lr_mult = (param_id < num_spec_params ? 
                     param.param(param_idx).lr_mult() : 1);
    if (param_name.size() == 0) {
      ostringstream os;
      os << "layer" << layer_index << "_param" << param_idx; 
      param_name = os.str();
    }
    vector<shared_ptr<Tensor<Dtype>>>& layer_tensors = layers_.back()->tensors();
    // If shared params, release the memory for duplicate params.
    if (param_idx < num_spec_params && name_to_learnable_index_.count(name) > 0) {
      for (int tensor_idx = 0; tensor_idx < layer_tensors.size(); ++tensor_idx) {
        layer_tensors[tensor_idx].reset();
      }
    }
    // If not shared, push tensors to learnable_params_ and 
    // mark the name for these params.
    else {
      for (int tensor_idx = 0; tensor_idx < layer_tensors.size(); ++tensor_idx) {
        name_to_learnable_index_[param_name] = learnable_params_.size();
        learnable_params_.push_back(layer_tensors[tensor_idx].get());
        params_weight_decay_.push_back(decay_mult);
        params_lr_.push_back(lr_mult);
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardBackward() {
  Dtype loss;
  // Forward and compute loss.
  for (int i = 0; i < layers_.size(); ++i) {
    Dtype layer_loss = layers_[i]->Forward(
        bottom_tensors_[i], top_tensors_[i]);
    loss += layer_loss;
  }
  // Backward and compute gradient.
  for (int i = layers_.size() - 1; i >= 0; --i) {
    layers_[i]->Backward(top_tensors_[i], bottom_tensors_[i]);
  }
  return loss;
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

INSTANTIATE_CLASS(Net);

} // namespace lich