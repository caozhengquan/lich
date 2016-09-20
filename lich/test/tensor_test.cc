#include "lich/src/tensor.h"
#include "lich/test/test_main.h"

#include <gtest/gtest.h>

namespace lich {

template <typename Dtype>
class TensorSimpleTest : public ::testing::Test {
 protected:
  TensorSimpleTest()
      : tensor_(new Tensor<Dtype>()), 
        tensor_preshaped_(new Tensor<Dtype>({2, 3, 4, 5})) {}
  ~TensorSimpleTest() {
    delete tensor_;
    delete tensor_preshaped_;
  }

  Tensor<Dtype>* const tensor_;
  Tensor<Dtype>* const tensor_preshaped_;
};

TYPED_TEST_CASE(TensorSimpleTest, TestDtypes);

TYPED_TEST(TensorSimpleTest, TestInitialized) {
  EXPECT_TRUE(this->tensor_);
  EXPECT_TRUE(this->tensor_preshaped_);
  EXPECT_EQ(this->tensor_->count(), 0);
  EXPECT_EQ(this->tensor_preshaped_->count(), 120);
  EXPECT_EQ(this->tensor_->num_axes(), 0);
  EXPECT_EQ(this->tensor_preshaped_->num_axes(), 4);
  EXPECT_EQ(this->tensor_preshaped_->CanonicalAxisIndex(-1), 3);
}

TYPED_TEST(TensorSimpleTest, TestDataPointer) {
  EXPECT_EQ(this->tensor_->data(), nullptr);
  EXPECT_NE(this->tensor_preshaped_->data(), nullptr);
  EXPECT_EQ(this->tensor_->diff(), nullptr);
  EXPECT_NE(this->tensor_preshaped_->diff(), nullptr);
}

TYPED_TEST(TensorSimpleTest, TestReshape) {
  this->tensor_->Reshape({2, 3, 4, 5});
  EXPECT_EQ(this->tensor_->count(), this->tensor_preshaped_->count());
  EXPECT_EQ(this->tensor_->num_axes(), this->tensor_preshaped_->num_axes());
  EXPECT_EQ(this->tensor_->CanonicalAxisIndex(-1),
           this->tensor_preshaped_->CanonicalAxisIndex(-1));
  EXPECT_TRUE(this->tensor_->data());
  EXPECT_TRUE(this->tensor_->diff());
}

} // namespace lich