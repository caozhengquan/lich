#include "lich/lib/math.h"
#include "lich/src/tensor.h"
#include "lich/src/test_main.h"
#include <gtest/gtest.h>

namespace lich {

template <typename Dtype>
class MathTest : public ::testing::Test {
  protected:
   MathTest() : bottom_(new Tensor<Dtype>()),
                top_(new Tensor<Dtype>()) {}
   ~MathTest() {
     delete bottom_;
     delete top_;
   }
   
   Tensor<Dtype>* bottom_;
   Tensor<Dtype>* top_;
};

}