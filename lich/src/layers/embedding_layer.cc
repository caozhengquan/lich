#include "lich/src/filler.h"
#include "lich/src/layer_factory.h"
#include "lich/src/layers/embedding_layer.h"

namespace lich {

template <typename Dtype>
void EmbeddingLayer<Dtype>::LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
                                       const vector<Tensor<Dtype>*>& top) {
  this->tensors_.resize(1);
  this->tensors_[0].reset(new Tensor<Dtype>({input_dim_, embed_dim_}));
  shared_ptr<Filler<Dtype>> weight_filler(Filler<Dtype>::GetFiller(
      this->layer_param_.embedding_param().weight_filler()));
  weight_filler->Fill(this->tensors_[0].get());
}

template <typename Dtype>
void EmbeddingLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                    const vector<Tensor<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(embed_dim_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EmbeddingLayer<Dtype>::ForwardCpu(const vector<Tensor<Dtype>*>& bottom,
                                       const vector<Tensor<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* weight = this->tensors_[0]->data();
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* top_data = top[0]->mutable_data();
  // Forward just copy the corresponding embedding to top data.
  for (int i = 0; i < count; ++i) {
    int embed_idx = bottom_data[i];
    lich_copy(embed_dim_, weight + embed_idx * embed_dim_, 
              top_data + i * embed_dim_);
  }
}

template <typename Dtype>
void EmbeddingLayer<Dtype>::BackwardCpu(const vector<Tensor<Dtype>*>& top,
                                        const vector<Tensor<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->diff();
  const Dtype* bottom_data = bottom[0]->data();
  Dtype* weight_diff = this->tensors_[0]->mutable_diff();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    // Update weight gradient: gradient = bottom_data * delta(l+1)
    int embed_idx = bottom_data[i];
    lich_axpy<Dtype>(embed_dim_, Dtype(1), top_diff + i * embed_dim_,
                     weight_diff + embed_idx * embed_dim_);
  }
}

INSTANTIATE_CLASS(EmbeddingLayer);
REGISTER_LAYER(Embedding);

} // namespace lich