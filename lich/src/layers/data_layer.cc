#include "lich/src/layers/data_layer.h"
#include "lich/src/common.h"
#include "lich/src/layer_factory.h"

#include <iostream>
#include <sstream>

namespace lich {

namespace {

template <typename Dtype>
vector<Dtype> ReadLineRepeated(std::ifstream& input_stream) {
  string line;
  if (std::getline(input_stream, line)) {
    vector<Dtype> input_values;
    std::istringstream in(line);
    Dtype value;
    while (in >> value) {
      input_values.push_back(value);
    }
    return input_values;
  }
  else {
    input_stream.clear();
    input_stream.seekg(0, std::ios::beg);
    return ReadLineRepeated<Dtype>(input_stream);
  }
}

} // namespace

template <typename Dtype>
void DataLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                               const vector<Tensor<Dtype>*>& top) {
  // Data of features.
  DataParameter* data_param = this->layer_param_.mutable_data_param();
  CHECK_EQ(data_param->output_dim_size(), this->layer_param_.top_size());
  for (int top_id = 0; top_id < this->layer_param_.top_size(); ++top_id) {
    top[top_id]->Reshape({batch_size_, data_param->output_dim(top_id)});
  }
}

template <typename Dtype>
void DataLayer<Dtype>::NextBatch(const vector<Tensor<Dtype>*>& top) {
  DataParameter data_param = this->layer_param_.data_param();
  const int output_dim_size = data_param.output_dim_size();
  CHECK_EQ(output_dim_size, top.size());

  // Check the count of every top tensor.
  int feature_count = 0;
  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    CHECK_EQ(top[top_idx]->count(), batch_size_ * data_param.output_dim(top_idx));
    feature_count += data_param.output_dim(top_idx);
  }

  // Assign the file data to top tensors.
  for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    vector<Dtype> input_values = ReadLineRepeated<Dtype>(input_stream_);
    CHECK_GE(input_values.size(), feature_count);
    int batch_input_idx = 0;
    for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
      Dtype* data = top[top_idx]->mutable_data();
      for (int data_idx = 0; data_idx < data_param.output_dim(top_idx); ++data_idx) {
        int actual_data_idx = data_param.output_dim(top_idx) * batch_idx + data_idx;
        int actual_input_idx = batch_input_idx + data_idx;
        data[actual_data_idx] = input_values[actual_input_idx];
      }
      batch_input_idx += data_param.output_dim(top_idx);
    }
  }
} 

template <typename Dtype>
void DataLayer<Dtype>::ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                                  const vector<Tensor<Dtype>*>& top) {
  NextBatch(top);
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER(Data);

} // namespace lich