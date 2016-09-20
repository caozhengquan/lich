#ifndef LICH_SRC_LAYER_FACTORY_H_
#define LICH_SRC_LAYER_FACTORY_H_

#include "lich/src/layer.h"
#include "lich/src/common.h"
#include "lich/proto/layer_param.pb.h"

#include <functional>
#include <unordered_map>
#include "glog/logging.h"

namespace lich {

template <typename Dtype>
class LayerRegistry {
 public:
  LayerRegistry() {}
  using Creator = std::function<Layer<Dtype>*(const LayerParameter&)>;

  static LayerRegistry* Global() {
    static LayerRegistry* registry = new LayerRegistry;
    return registry;
  }

  void Register(const string& type, Creator creator) {
    if (registry_.count(type)) {
      CHECK(false) << "Layer type " << type << "already exists.";
    }
    registry_[type] = creator;
  }
  
  Layer<Dtype>* CreateLayer(const LayerParameter& param) {
    const string& type = param.type();
    CHECK_EQ(registry_.count(type), 1) << "Unkown layer type: " << type
        << ", registered types: " << RegisteredTypes();
    return registry_[type](param);
  }

  string RegisteredTypes() {
    string types;
    for (auto pair : registry_) {
      types += (pair.first + " ");
    }
    return types;
  }

 private:
  mutable std::unordered_map<string, Creator> registry_;

  DISALLOW_COPY_AND_ASSIGN(LayerRegistry);
};

namespace register_layer {

template<typename Dtype>
class LayerRegisterHelper {
 public:
  LayerRegisterHelper(const string& type, 
                      typename ::lich::LayerRegistry<Dtype>::Creator creator) {
    LayerRegistry<Dtype>::Global()->Register(type, creator);
  }
};

} // namespace register_layer

#define REGISTER_LAYER(type)                                                      \
  template <typename Dtype>                                                       \
  Layer<Dtype>* type##Layer_creator(const LayerParameter& param)                  \
  {                                                                               \
    return new type##Layer<Dtype>(param);                                         \
  }                                                                               \
  static ::lich::register_layer::LayerRegisterHelper<float> g_creator_f_##type(#type, type##Layer_creator<float>); \
  static ::lich::register_layer::LayerRegisterHelper<double> g_creator_d_##type(#type, type##Layer_creator<double>)  

} // namespace lich

#endif