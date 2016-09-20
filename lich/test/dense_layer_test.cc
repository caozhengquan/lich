#include "lich/src/tensor.h"
#include "lich/src/layers/dense_layer.h"
#include "lich/test/test_main.h"
#include "lich/src/layer_factory.h"

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
  LayerParameter layer_param;
  layer_param.set_type("Dense");
  DenseParameter* dense_param = layer_param.mutable_dense_param();
  dense_param->set_num_output(10);
  dense_param->set_axis(1);
  FillerParameter* filler_param = dense_param->mutable_weight_filler();
  filler_param->set_type("constant");
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(layer_param));
  layer->SetUp(this->bottom_tensor_vec_, this->top_tensor_vec_);
  EXPECT_EQ(this->top_tensor_->shape(), vector<int>({2, 10}));
}

} // namespace lich