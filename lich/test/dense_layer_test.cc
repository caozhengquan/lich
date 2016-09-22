#include "lich/src/tensor.h"
#include "lich/src/layers/dense_layer.h"
#include "lich/test/test_main.h"
#include "lich/src/layer_factory.h"
#include "lich/test/gradient_checker.h"
#include "lich/src/layer_param_builder.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

namespace lich {

template <typename Dtype>
class DenseLayerTest : public ::testing::Test {
 protected:
  DenseLayerTest() 
      : bottom_tensor_(new Tensor<Dtype>({2, 3, 4, 5})),
        top_tensor_(new Tensor<Dtype>()) {
    top_tensor_vec_.push_back(top_tensor_);
    FillerParameter filler = FillerParameterBuilder().type("uniform")
                            .min(0).max(1).Build();
    layer_param_ = LayerParameterBuilder().type("Dense")
                  .dense_param(10, true, filler, filler, 1)
                  .Build();
  }
  ~DenseLayerTest() {
    delete bottom_tensor_;
    delete top_tensor_;
  }
 
  Tensor<Dtype>* bottom_tensor_;
  Tensor<Dtype>* top_tensor_;
  vector<Tensor<Dtype>*> bottom_tensor_vec_;
  vector<Tensor<Dtype>*> top_tensor_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(DenseLayerTest, TestDtypes);
TYPED_TEST(DenseLayerTest, TestSetUp) {
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  EXPECT_EQ(this->top_tensor_->shape(), vector<int>({2, 10}));
}

TYPED_TEST(DenseLayerTest, TestBackward) {
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  this->bottom_tensor_vec_.push_back(this->bottom_tensor_);
  GradientChecker<TypeParam> grad_checker;
  grad_checker.CheckGradient(layer.get(), this->bottom_tensor_vec_,
                             this->top_tensor_vec_);
}

} // namespace lich