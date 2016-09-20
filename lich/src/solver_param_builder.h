#ifndef LICH_SRC_SOLVER_PARAM_BUILDER_H_
#define LICH_SRC_SOLVER_PARAM_BUILDER_H_

#include "lich/proto/solver_param.pb.h"

namespace lich {

template <typename>
class SolverParamBuilder {
 public:
  SolverParamBuilder() {}

  SolverParamBuilder& base_lr(float base_lr) {
    solver_param_.set_base_lr(base_lr);
    return *this;
  }

  SolverParamBuilder& max_iter(int max_iter) {
    solver_param_.set_max_iter(max_iter);
    return *this;
  }
  
  SolverParamBuilder& iter_size(int iter_size) {
    solver_param_.set_iter_size(iter_size);
    return *this;
  } 

  SolverParamBuilder& type(string type) {
    solver_param_.set_type(type);
    return *this;
  }

  SolverParamBuilder& regularization_type(string regularization_type) {
    solver_param_.set_regularization_type(regularization_type);
    return *this;
  }

  SolverParamBuilder& weight_decay(float weight_decay) {
    solver_param_.set_weight_decay(weight_decay);
    return *this;
  }

  SolverParamBuilder& momentum(float momentum) {
    solver_param_.set_momentum(momentum);
    return *this;
  }
  
  SolverParameter Build() {
    return solver_param_;
  }
  
 private:
  SolverParameter solver_param_;
};

} // namespace lich 


#endif