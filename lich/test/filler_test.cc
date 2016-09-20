#include "lich/src/filler.h"
#include "lich/test/test_main.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

namespace lich {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest()
      : tensor_(new Tensor<Dtype>({2, 3, 4, 5})),
        filler_param_() {
    filler_param_.set_type("constant");
    filler_param_.set_value(10.);
    filler_ = Filler<Dtype>::GetFiller(filler_param_);
    filler_->Fill(tensor_);
  }
  ~ConstantFillerTest() {
    delete tensor_;
    delete filler_;
  }

  Tensor<Dtype>* const tensor_;
  FillerParameter filler_param_;
  Filler<Dtype>* filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);
TYPED_TEST(ConstantFillerTest, TestFill) {
  ASSERT_TRUE(this->tensor_);
  ASSERT_TRUE(this->filler_);
  const int count = this->tensor_->count();
  const TypeParam* data = this->tensor_->data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], this->filler_param_.value());
  }
}

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest()
      : tensor_(new Tensor<Dtype>({2, 3, 4, 5})),
        filler_param_() {
    filler_param_.set_type("guassian");
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_ = Filler<Dtype>::GetFiller(filler_param_);
    filler_->Fill(tensor_);
  }
  virtual ~GaussianFillerTest() { 
    delete tensor_;
    delete filler_;
  }
  Tensor<Dtype>* const tensor_;
  FillerParameter filler_param_;
  Filler<Dtype>*  filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);
TYPED_TEST(GaussianFillerTest, TestFill) {
  // Mean and var of generated data should be around the
  // true mean and std.
  ASSERT_TRUE(this->tensor_);
  ASSERT_TRUE(this->filler_);
  const int count = this->tensor_->count();
  const TypeParam* data = this->tensor_->data();
  TypeParam mean(0);
  TypeParam var(0);
  for (int i = 0; i < count; ++i) {
    mean += data[i];
    var += (data[i] - this->filler_param_.mean()) *
        (data[i] - this->filler_param_.mean());
  }
  mean /= count;
  var /= count;
  EXPECT_GE(mean, this->filler_param_.mean() - this->filler_param_.std() * 5);
  EXPECT_LE(mean, this->filler_param_.mean() + this->filler_param_.std() * 5);
  TypeParam target_var = this->filler_param_.std() * this->filler_param_.std();
  EXPECT_GE(var, target_var / 5.);
  EXPECT_LE(var, target_var * 5.);
}

} // namespace lich