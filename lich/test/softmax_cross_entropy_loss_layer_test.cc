#include "lich/src/layers/softmax_cross_entropy_loss_layer.h"
#include "lich/src/layer_param_builder.h"
#include "lich/test/test_main.h"
#include "lich/test/gradient_checker.h"

#include <gtest/gtest.h>

namespace lich {

template <typename Dtype>
class SoftmaxCrossEntropyLossLayerTest : public ::testing::Test {
 protected:
  SoftmaxCrossEntropyLossLayerTest() 
      : bottom_tensor_(new Tensor<Dtype>({2, 4})),
        top_tensor_(new Tensor<Dtype>({2})) {
    top_tensor_vec_.push_back(top_tensor_);
    layer_param_ = LayerParameterBuilder().type("SoftmaxCrossEntropyLoss")
                  .softmax_param(1)
                  .Build();
  }
  ~SoftmaxCrossEntropyLossLayerTest() {
    delete bottom_tensor_;
    delete top_tensor_;
  }
 
  Tensor<Dtype>* bottom_tensor_;
  Tensor<Dtype>* top_tensor_;
  vector<Tensor<Dtype>*> bottom_tensor_vec_;
  vector<Tensor<Dtype>*> top_tensor_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(SoftmaxCrossEntropyLossLayerTest, TestDtypes);
TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestSetUp) {
  Tensor<TypeParam>* label_tensor = new Tensor<TypeParam>({2, 1});
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  this->bottom_tensor_vec_.push_back(label_tensor);
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  EXPECT_EQ(this->top_tensor_->shape(), vector<int>());
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestForward) {
  // Set the dummy input to bottom tensor.
  Tensor<TypeParam>* label_tensor = new Tensor<TypeParam>({2, 1});
  label_tensor->mutable_data()[0] = 2;
  label_tensor->mutable_data()[1] = 1;
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  this->bottom_tensor_vec_.push_back(label_tensor);
  vector<int> dummy_input = {1, 2, 3, 4, 1, 2, 2, 1};
  for (int idx = 0; idx < this->bottom_tensor_->count(); ++idx) {
    this->bottom_tensor_->mutable_data()[idx] = dummy_input[idx];
  }
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  layer->Forward(this->bottom_tensor_vec_, this->top_tensor_vec_);
  EXPECT_NEAR(this->top_tensor_vec_[0]->data()[0],
              1.2232992833196819, 1e-4);
}


TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestBackward) {
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  Tensor<TypeParam>* label_tensor = new Tensor<TypeParam>({2, 1});
  label_tensor->mutable_data()[0] = 2;
  label_tensor->mutable_data()[1] = 1;
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  this->bottom_tensor_vec_.push_back(label_tensor);
  GradientChecker<TypeParam> grad_checker;
  grad_checker.CheckGradient(layer.get(), this->bottom_tensor_vec_,
                             this->top_tensor_vec_, 0);
}


} // namespace lich