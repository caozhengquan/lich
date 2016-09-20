#include "lich/src/tensor.h"
#include "lich/src/layers/data_layer.h"
#include "lich/test/test_main.h"
#include "lich/src/layer_factory.h"
#include "lich/src/layer_param_builder.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

namespace lich {

template <typename Dtype>
class EmbeddingLayerTest : public ::testing::Test {
 protected:
  EmbeddingLayerTest() 
      : bottom_tensor_(new Tensor<Dtype>({2, 4})),
        top_tensor_(new Tensor<Dtype>()) {
    bottom_tensor_vec_.push_back(bottom_tensor_);
    top_tensor_vec_.push_back(top_tensor_);
    FillerParameter filler_param = FillerParameterBuilder<Dtype>()
        .type("uniform").min(0).max(1).Build();
    layer_param_ = LayerParamterBuilder().type("Embedding")
                  .bottom({"bottom"})
                  .top({"top"})
                  .embedding_param(5, 10, filler_param)
                  .Build();
  }

  ~EmbeddingLayerTest() {
    delete bottom_tensor_;
    delete top_tensor_;
  }
 
  Tensor<Dtype>* bottom_tensor_;
  Tensor<Dtype>* top_tensor_;
  vector<Tensor<Dtype>*> bottom_tensor_vec_;
  vector<Tensor<Dtype>*> top_tensor_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(EmbeddingLayerTest, TestDtypes);
TYPED_TEST(EmbeddingLayerTest, TestSetUp) {
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<Dtype>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  EXPECT_EQ(this->top_tensor_vec_[0]->shape(), vector<int>({2, 4, 10}));
  vector<shared_ptr<Tensor<TypeParam>>> tensors = layer->tensors();
  EXPECT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0]->shape(), vector<int>({5, 10}));
}

TYPED_TEST_CASE(EmbeddingLayerTest, TestDtypes);
TYPED_TEST(EmbeddingLayerTest, TestForward) {
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<Dtype>::Global()->CreateLayer(this->layer_param_));
  vector<int> dummy_input = {0, 1, 2, 3, 3, 2, 1, 0};
  const int count = this->bottom_tensor_->count();

  TypeParam* mutable_bottom_data = this->bottom_tensor_->mutable_data();
  for (int i = 0; i < count; ++i) {
    mutable_bottom_data[i] = dummy_input[i];
  }
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  layer->Forward(this->bottom_tensor_vec_, this->top_tensor_vec_);
  
  const TypeParam* bottom_data = this->bottom_tensor_->data();
  const TypeParam* top_data = this->top_tensor_->data();
  shared_ptr<Tensor<TypeParam>> weight = layer->tensors()[0];
  const TypeParam* weight_data = weight->data();
  for (int i = 0; i < count; ++i) {
    int index = bottom_data[i];
    for (int j = 0; j < 10; ++j) {
      LOG(INFO) << top_data[i * 10 + j];
      //EXPECT_EQ(weight_data[index * 10 + j],
      //          top_data[i * 10 + j]);
    }
  }
}


} // namespace lich