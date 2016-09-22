#ifndef LICH_TEST_GRADIENT_CHECKER_H_
#define LICH_TEST_GRADIENT_CHECKER_H_

#include "lich/src/tensor.h"
#include "lich/src/common.h"
#include "lich/src/layer.h"
#include "lich/test/gradient_checker.h"
#include <glog/logging.h>
#include <gtest/gtest.h>


namespace lich {

template <typename Dtype>
class GradientChecker {
 public:
  GradientChecker(const Dtype stepsize=1e-2, const Dtype threshold=1e-3,
                  const Dtype kink=0., const Dtype kink_range=-1)
      : stepsize_(stepsize), threshold_(threshold),
        kink_(kink), kink_range_(kink_range) {}
  
  // If check_bottom == -1, then check all bottom tensors. (common case)
  // If check_bottom == -2, then check none bottom tensors. (such as embedding layer)
  // If check_bottom >= 0, then check only the check_bottom index. (such as loss layer)
  void CheckGradient(
      Layer<Dtype>* layer, const vector<Tensor<Dtype>*>& bottom,
      const vector<Tensor<Dtype>*>& top, int check_bottom=-1);

 private:
  Dtype stepsize_;
  Dtype threshold_;
  Dtype kink_;
  Dtype kink_range_;
  
  Dtype GetLossAndGradient(const vector<Tensor<Dtype>*> top);
  Dtype GetEstimatedGradient(
      Layer<Dtype>* layer, const vector<Tensor<Dtype>*>& bottom,
      const vector<Tensor<Dtype>*>& top, 
      Tensor<Dtype>* check_tensor, int input_id);
};

namespace {

// Restore every diff in tensors_need_check to computed_gradient_tensors
template <typename Dtype>
void RestoreComputedGradient(
    vector<shared_ptr<Tensor<Dtype>>>& computed_gradient_tensors,
    const vector<Tensor<Dtype>*>& tensors_need_check) {
  for (int idx = 0; idx < tensors_need_check.size(); ++idx) {
    Tensor<Dtype>* tensor = tensors_need_check[idx];
    computed_gradient_tensors[idx].reset(new Tensor<Dtype>(tensor->shape()));
    const int count = tensor->count();
    const Dtype* tensor_diff = tensor->diff();
    Dtype* computed_grad = computed_gradient_tensors[idx]->mutable_data();
    lich_copy(count, tensor_diff, computed_grad);
  }
}

} // namespace

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradient(
    Layer<Dtype>* layer, const vector<Tensor<Dtype>*>& bottom,
    const vector<Tensor<Dtype>*>& top, int check_bottom) {
  layer->SetUp(bottom, top);  
  // Restore the tensors which need to check.
  vector<Tensor<Dtype>*> tensors_need_check;
  for (int idx = 0; idx < layer->tensors().size(); ++idx) {
    tensors_need_check.push_back(layer->tensors()[idx].get());
  }
  if (check_bottom != -2) {
    for (int idx = 0; idx < bottom.size(); ++idx) {
      if (check_bottom == -1 || idx == check_bottom) {
        tensors_need_check.push_back(bottom[idx]);
      }
    }
  }

  CHECK_GT(tensors_need_check.size(), 0) << "No tensors to check.";
  // Forward Backward to restore the computed gradient for the tensors
  // which need to check.
  layer->Forward(bottom, top);
  GetLossAndGradient(top);
  layer->Backward(top, bottom);
  vector<shared_ptr<Tensor<Dtype>>> computed_gradient_tensors(
      tensors_need_check.size());
  RestoreComputedGradient(computed_gradient_tensors, tensors_need_check);

  for (int tensor_id = 0; tensor_id < tensors_need_check.size(); ++tensor_id) {
    Tensor<Dtype>* check_tensor = tensors_need_check[tensor_id];
    const Dtype* computed_gradients = computed_gradient_tensors[tensor_id]->data();
    for (int input_id = 0; input_id < check_tensor->count(); ++input_id) {
      Dtype estimated_grad = 
        GetEstimatedGradient(layer, bottom, top, check_tensor, input_id);
      Dtype computed_grad = computed_gradients[input_id];
      Dtype feature = check_tensor->data()[input_id];
      if (kink_ - kink_range_ > fabs(feature) || 
          fabs(feature) > kink_ + kink_range_) {
        // Compare by relative error, but for very small values, compare 
        // by absolute error (by set scale to 1)
        Dtype max_grad = std::max(fabs(computed_grad), fabs(estimated_grad));
        Dtype scale = std::max(max_grad, Dtype(1));
        EXPECT_NEAR(computed_grad, estimated_grad, threshold_ * scale);
      }
    }
  }
}

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetLossAndGradient(const vector<Tensor<Dtype>*> top) {
  Dtype loss = 0;
  for (int idx = 0; idx < top.size(); ++idx) {
    const Dtype* top_data = top[idx]->data();
    Dtype* top_diff = top[idx]->mutable_diff();
    const int count = top[idx]->count();
    for (int data_id = 0; data_id < count; ++data_id) {
      loss += top_data[data_id] * top_data[data_id];
    }
    // d(loss) / d(yi) = yi
    lich_copy(count, top_data, top_diff);
  }
  loss /= 2.0;
  return loss;
}

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetEstimatedGradient(
    Layer<Dtype>* layer, const vector<Tensor<Dtype>*>& bottom,
    const vector<Tensor<Dtype>*>& top, 
    Tensor<Dtype>* check_tensor, int input_id) {
  Dtype positive_loss = 0;
  Dtype negative_loss = 0;
  Dtype gradient = 0;
  check_tensor->mutable_data()[input_id] += stepsize_;
  layer->Forward(bottom, top);
  positive_loss = GetLossAndGradient(top);
  check_tensor->mutable_data()[input_id] -= 2 * stepsize_;
  layer->Forward(bottom, top);
  negative_loss = GetLossAndGradient(top);
  check_tensor->mutable_data()[input_id] += stepsize_;
  gradient = (positive_loss - negative_loss) / (2 * stepsize_);
  return gradient;
}

} // namespace lich

#endif