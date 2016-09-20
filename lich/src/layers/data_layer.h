#ifndef LICH_SRC_LAYERS_DATA_LAYER_H_
#define LICH_SRC_LAYERS_DATA_LAYER_H_

#include "lich/src/layer.h"
#include "lich/src/tensor.h"
#include <glog/logging.h>

#include <fstream>

namespace lich {
  
template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param) 
      : Layer<Dtype>(param), batch_size_(param.data_param().batch_size()),
        input_stream_(param.data_param().source(), std::ifstream::in) {
    CHECK_EQ(input_stream_.good(), true) << "Error while opening file "
        << param.data_param().source();
  }

 protected:
  int batch_size_;
  std::ifstream input_stream_;

  virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
                       const vector<Tensor<Dtype>*>& top) override;

  virtual void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                          const vector<Tensor<Dtype>*>& top) override;
                           
  virtual void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                           const vector<Tensor<Dtype>*>& bottom) override {}   

  void NextBatch(const vector<Tensor<Dtype>*>& top);
};

} // namespace lich

#endif