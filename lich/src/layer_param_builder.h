#ifndef LICH_SRC_LAYER_BUILDER_H_
#define LICH_SRC_LAYER_BUILDER_H_

#include "lich/src/layer_factory.h"
#include "lich/src/layer.h"
#include "lich/proto/filler_param.pb.h"
#include "lich/proto/layer_param.pb.h"

namespace lich {

template <typename Dtype>
class FillerParameterBuilder {
 public:
  FillerParameterBuilder() {}
  
  FillerParameterBuilder& type(string type) {
    filler_param_.set_type(type);
    return *this;
  }

  FillerParameterBuilder& value(float value) {
    filler_param_.set_value(value);
    return *this;
  }

  FillerParameterBuilder& min(float min) {
    filler_param_.set_min(min);
    return *this;
  }

  FillerParameterBuilder& max(float max) {
    filler_param_.set_max(max);
    return *this;
  }

  FillerParameterBuilder& mean(float mean) {
    filler_param_.set_mean(mean);
    return *this;
  }

  FillerParameterBuilder& std(float std) {
    filler_param_.set_std(std);
    return *this;
  }

  FillerParameter Build() { return filler_param_; }

 private:
  FillerParameter filler_param_;
};

template <typename Dtype>
class LayerParameterBuilder {
 public:
  LayerParameter() {}
  
  LayerParameterBuilder& name(string name) {
    layer_param_.set_name(name);
    return *this;
  } 

  LayerParameterBuilder& type(string type) {
    layer_param_.set_type(type);
    return *this;
  }

  LayerParameterBuilder& bottom(vector<string> bottoms) {
    for (string bottom_name : bottoms) {
      layer_param_.add_bottom(bottom_name);
    }
    return *this;
  }
  
  LayerParameterBuilder& top(vector<string> tops) {
    for (string top_name : tops) {
      layer_param_.add_top(top_name);
    }
    return *this;
  }

  LayerParameterBuilder& data_param(string source, int batch_size,
                           vector<int> output_dim) {
    DataParameter* data_param = this->layer_param_.mutable_data_param();
    data_param->set_source(source);
    data_param->set_batch_size(batch_size);
    for (int dim : output_dim) {
      data_param->add_output_dim(dim);
    }
    return *this;
  }

  LayerParameterBuilder& embedding_param(int input_dim, int embed_dim, FillerParameter weight_filler) {
    EmbeddingParameter* embed_param = this->layer_param_.mutable_embedding_param();
    embed_param->set_input_dim(input_dim);
    embed_param->set_embed_dim(embed_dim);
    FillerParameter* filler_param = embed_param->mutable_weight_filler();
    filler_param->CopyFrom(weight_filler);
    return *this;
  }

  LayerParameterBuilder& dense_param(
      int num_output, bool bias_term, FillerParameter weight_filler,
      FillerParameter bias_filler, int axis) {
    DenseParameter* dense_param = this->layer_param_.mutable_dense_param();
    dense_param->set_num_output(num_output);
    dense_param->set_bias_term(bias_term);
    dense_param->mutable_weight_filler()->CopyFrom(weight_filler);
    dense_param->mutable_bias_filler()->CopyFrom(bias_filler);
    dense_param->set_axis(axis);
    return *this;
  } 
  
  LayerParameterBuilder& softmax_param(int axis) {
    SoftmaxParameter* softmax_param = this->layer_param_.mutable_softmax_param();
    softmax_param->set_axis(axis);
    return *this;
  }

  LayerParameterBuilder Build() {
    return layer_param_;
  }

 protected:
  LayerParameter layer_param_;
};

} // namespace lich

#endif