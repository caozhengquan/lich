#include "lich/src/tensor.h"
#include "lich/src/layers/data_layer.h"
#include "lich/test/test_main.h"
#include "lich/src/layer_factory.h"
#include "lich/src/layer_param_builder.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

namespace lich {

template <typename Dtype>
class DataLayerTest : public ::testing::Test {
 protected:
  DataLayerTest() 
      : data_tensor_(new Tensor<Dtype>()),
        label_tensor_(new Tensor<Dtype>()) {
    top_tensor_vec_.push_back(data_tensor_);
    top_tensor_vec_.push_back(label_tensor_);
    layer_param_ = LayerParameterBuilder().type("Data")
                   .top({"data", "label"})
                   .data_param("../lich/test/dummy_data.txt", 2, {4, 1})
                   .Build();
  }
  ~DataLayerTest() {
    delete data_tensor_;
    delete label_tensor_;
  }
 
  Tensor<Dtype>* data_tensor_;
  Tensor<Dtype>* label_tensor_;
  vector<Tensor<Dtype>*> top_tensor_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(DataLayerTest, TestDtypes);
TYPED_TEST(DataLayerTest, TestForward) {
  shared_ptr<Layer<TypeParam>> layer(
      LayerRegistry<TypeParam>::Global()->CreateLayer(this->layer_param_));
  layer->SetUp(this->top_tensor_vec_, this->top_tensor_vec_);
  for (int k = 0; k < 2; ++k) {
    layer->Forward(this->top_tensor_vec_, this->top_tensor_vec_);
    const TypeParam* data = this->top_tensor_vec_[0]->data();
    const int data_count = this->top_tensor_vec_[0]->count();
    const TypeParam* label = this->top_tensor_vec_[1]->data();
    const int label_count = this->top_tensor_vec_[1]->count();
    for (int i = 0; i < data_count; ++i) {
      EXPECT_EQ(i, data[i]);
    }
    for (int i = 0; i < label_count; ++i) {
      EXPECT_EQ(i, label[i]);
    }
  }
}

} // namespace lich