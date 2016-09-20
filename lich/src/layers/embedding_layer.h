#ifndef LICH_SRC_LAYERS_EMBEDDING_LAYER_H_
#define LICH_SRC_LAYERS_EMBEDDING_LAYER_H_

#include "lich/src/layer.h"
#include "lich/src/tensor.h"

namespace lich {

template <typename Dtype>
class EmbeddingLayer : public Layer<Dtype> {
 public:
  EmbeddingLayer(const LayerParameter& param) : Layer<Dtype>(param) {
    EmbeddingParameter* embed_param = this->layer_param_.mutable_embedding_param();
    input_dim_ = embed_param->input_dim();
    embed_dim_ = embed_param->embed_dim();
    CHECK_GT(input_dim_, 0) << "input_dim must be positive for embedding";
    CHECK_GT(embed_dim_, 0) << "embed_dim must be positive for embedding";
  }

 protected:
  int input_dim_;
  int embed_dim_;

  void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
                  const vector<Tensor<Dtype>*>& top) override;

  void Reshape(const vector<Tensor<Dtype>*>& bottom,
               const vector<Tensor<Dtype>*>& top) override;

  void ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                  const vector<Tensor<Dtype>*>& top) override;

  void BackwardCpu(const vector<Tensor<Dtype>*>& top,
                   const vector<Tensor<Dtype>*>& bottom) override;
};

} // namespace lich

#endif