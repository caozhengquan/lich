#ifndef LICH_SRC_TENSOR_H_
#define LICH_SRC_TENSOR_H_

#include "lich/src/common.h"
#include "lich/lib/macros.h"
#include <glog/logging.h>

#include <vector>

namespace lich {
    
template <typename Dtype>
class Tensor {
 public:
  // Constructors
  Tensor() : data_(nullptr), diff_(nullptr), count_(0), capacity_(0) {}
  explicit Tensor(const std::vector<int>& shape);
  
  void Reshape(const std::vector<int>& shape);
  void ReshapeLike(const Tensor<Dtype>& rhs) {
    vector<int> shape = rhs.shape();
    this->Reshape(shape);
  }
  void Update();
  
  int count() const { return count_; }
  int count(int start_axis, int end_axis) const {
    CHECK_GE(start_axis, 0);
    CHECK_LT(start_axis, num_axes());
    CHECK_GE(end_axis, start_axis);
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int idx = start_axis; idx < end_axis; ++idx) {
      count *= shape_[idx];
    }
    return count;
  }
  int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  vector<int> shape() const { return shape_; }
  int shape(int axis) const {
    CHECK_GE(axis, 0);
    CHECK_LT(axis, num_axes());
    return shape_[axis];
  }

  int num_axes() const { return shape_.size(); }
  const Dtype* data() const;
  const Dtype* diff() const;
  Dtype* mutable_data();
  Dtype* mutable_diff();
  
  int CanonicalAxisIndex(int index) const {
    CHECK_GE(index, -num_axes());
    CHECK_LT(index, num_axes());
    if (index < 0) {
      return index + num_axes();
    }
    return index;
  }
  
 private:
  // Restore the data for a tensor.
  shared_ptr<void> data_;
  // For loss layer, restore loss_weight.
  // Otherwise, restore the error which need to back propagation.
  shared_ptr<void> diff_;
  vector<int> shape_;
  int count_;
  int capacity_;
  
  DISALLOW_COPY_AND_ASSIGN(Tensor);
};

} // namespace lich
#endif