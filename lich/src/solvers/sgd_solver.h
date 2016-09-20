#ifndef LICH_SRC_SOLVERS_SGD_SOLVER_H_
#define LICH_SRC_SOLVERS_SGD_SOLVER_H_

#include "lich/src/common.h"
#include "lich/src/tensor.h"

namespace lich {

template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  SGDSolver(const SolverParameter& param, Net<Dtype>* net) 
      : Solver(param, net) {
    Prepare();
  }
 
 protected:
  // History restored gradients for learnable params in the net_
  vector<shared_ptr<Tensor<Dtype>>> history_;

  virtual void Prepare();
  virtual void ApplyUpdate() override;
  virtual Dtype GetLearningRate();

  // Normalize the value of learnable params by iter size given param id.
  // If the iter size is equal to 1, do nothing.
  virtual void Normalize(int param_id);
  // Regularize the learnable params by specific regularization type.
  virtual void Regularize(int param_id);
  // Update the gradient if needed, such as momentum, etc.
  virtual void UpdateGradient(int param_id, Dtype lr);

 private:
  DISALLOW_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
void SGDSolver<Dtype>::Prepare() {
  const vector<Tensor<Dtype>*>& net_params = net_->learnable_params();
  history_.clear();
  history_.resize(net_params.size());
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_[i] = shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape));
  }
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  return solver_param_.base_lr();
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  Dtype lr = GetLearningRate();
  for (int id = 0; id < net_->learnable_params().size(); ++id) {
    Normalize(id);
    Regularize(id);
    UpdateGradient(id, lr);
  }
  net_->Update();
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (solver_param_.iter_size() == 1) return;
  const Dtype alpha = Dtype(1) / solver_param_.iter_size();
  const int N = net_->learnable_params()[param_id]->count();
  Dtype* diff = net_->learnable_params()[param_id]->mutable_diff();
  lich_scal(N, alpha, diff);
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  string regularization_type = solver_param_.regularization_type();
  const Dtype solver_weight_decay = solver_param_.weight_decay();
  const Dtype param_weight_decay = net_->params_weight_decay()[param_id];
  const Dtype local_decay = solver_weight_decay * params_weight_decay;
  const int N = net_->learnable_params()[param_id]->count();
  const Dtype* data = net_->learnable_params()[param_id]->data();
  Dtype* diff = net_->learnable_params()[param_id]->mutable_diff();
  if (local_decay > 0) {
    if (regularization_type == "L2") {
      // gradient = decay * weight + gradient;
      lich_axpy(N, local_decay, data, diff);
    }
    else if (regularization_type == "L1") {
      Tensor<Dtype> temp({N});
      lich_sign(N, data, temp.mutable_data());
      // gradient = decay * (1 if weight > 0 else 0) + gradient;
      lich_axpy(N, local_decay, temp.data(), diff);
    }
    else {
      CHECK(false) << "regularization_type must be L1 or L2";
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::UpdateGradient(int param_id, Dtype lr) {
  const Dtype param_lr = net_->params_lr()[param_id];
  const Dtype local_lr = lr * param_lr;
  const Dtype momentum = solver_param_.momentum();
  const int N = net_->learnable_params()[param_id]->count();
  const Dtype* diff = net_->learnable_params()[param_id]->diff();
  // diff = momentum * diff + lr * cur_diff
  lich_axpby(N, local_lr, diff, momentum, history_[param_id]->mutable_data());
  lich_copy(N, history_[param_id]->data(),
            net_->learnable_params()[param_id]->mutable_diff());
}

} // namespace lich

#endif