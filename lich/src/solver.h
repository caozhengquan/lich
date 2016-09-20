#ifndef LICH_SRC_SOLVER_H_
#define LICH_SRC_SOLVER_H_

namespace lich {

template <typename Dtype>
class Solver {
 public:
  Solver<Dtype>::Solver(const SolverParameter& param, Net<Dtype>* net) 
    : solver_param_(param), net_(net) {}
  virtual ~Solver() {}

  virtual void Solve();

 protected:
  SolverParameter solver_param_;
  shared_ptr<Net<Dtype>> net_;

  virtual void ApplyUpdate() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(Solver);
};

template <typename Dtype>
void Solver<Dtype>::Solve() {
  const int max_iter = solver_param_.max_iter();
  const int iter_size = solver_param_.iter_size();
  for (int iter = 0; iter < max_iter; ++iter) {
    // Clear gradient for params and error for layers, since we
    // accumulate them in every step.
    Dtype iter_loss = 0;
    net_->ClearDiffError();
    for (int i = 0; i < solver_param_.iter_size; ++i) {
      Dtype loss = net_->ForwardBackWard();
      iter_loss += loss;
    }
    iter_loss /= iter_size;
    LOG(INFO) << "Iter " << iter << ", loss " << iter_loss;
    ApplyUpdate();
  }
}

} // namespace lich

#endif