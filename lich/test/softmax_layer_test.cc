#include "lich/src/layers/softmax_layer.h"
#include "lich/src/layer_param_builder.h"
#include "lich/test/test_main.h"
#include "lich/test/gradient_checker.h"

#include <gtest/gtest.h>

namespace lich {

template <typename Dtype>
class SoftmaxLayerTest : public ::testing::Test {
 protected:
  SoftmaxLayerTest() 
      : bottom_tensor_(new Tensor<Dtype>({2, 4})),
        top_tensor_(new Tensor<Dtype>()) {
    top_tensor_vec_.push_back(top_tensor_);
    layer_param_ = LayerParameterBuilder().type("Softmax")
                  .softmax_param(1)
                  .Build();
  }
  ~SoftmaxLayerTest() {
    delete bottom_tensor_;
    delete top_tensor_;
  }
 
  Tensor<Dtype>* bottom_tensor_;
  Tensor<Dtype>* top_tensor_;
  vector<Tensor<Dtype>*> bottom_tensor_vec_;
  vector<Tensor<Dtype>*> top_tensor_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypes);
TYPED_TEST(SoftmaxLayerTest, TestSetUp) {
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  ASSERT_EQ(this->top_tensor_->shape(), this->bottom_tensor_->shape());
}

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  // Set dummy input to bottom tensor.
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  vector<TypeParam> dummy_input = {1, 2, 3, 4, 1, 2, 2, 1};
  vector<TypeParam> dummy_output = {0.0320586, 0.08714432, 0.23688282, 0.64391426,
                                    0.13447071, 0.36552929, 0.36552929, 0.13447071};
  for (int idx = 0; idx < this->bottom_tensor_->count(); ++idx) {
    this->bottom_tensor_->mutable_data()[idx] = dummy_input[idx];
  }
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  layer->Forward(this->bottom_tensor_vec_, this->top_tensor_vec_);

  const TypeParam* data = this->top_tensor_->data();
  const int count = this->top_tensor_->count();
  for (int idx = 0; idx < count; ++idx) {
    EXPECT_NEAR(data[idx], dummy_output[idx], 1e-4);
  }
}

TYPED_TEST(SoftmaxLayerTest, TestBackward) {
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  vector<TypeParam> dummy_input = {1, 2, 3, 4, 1, 2, 2, 1};
  vector<TypeParam> dummy_output = {0.0320586, 0.08714432, 0.23688282, 0.64391426,
                                    0.13447071, 0.36552929, 0.36552929, 0.13447071};
  GradientChecker<TypeParam> grad_checker;
  grad_checker.CheckGradient(layer.get(), this->bottom_tensor_vec_,
                             this->top_tensor_vec_);
}


} // namespace lich