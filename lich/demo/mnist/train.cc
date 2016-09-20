#include "lich/src/lich.h"

share_ptr<Net<float>> BuildNet() {
  shared_ptr<Net<float>> net(new Net<float>);
  // FillerParameter
  FillerParameter weight_filler = FillerParameterBuilder().type("guassian")
      .mean(0).std(1).Build();
  FillerParameter bias_filler = FillerParameterBuilder().type("constant")
      .value(0).Build();
  // Data layer
  LayerParameter data_layer_param = LayerParameterBuilder().type("Data")
      .top({"data", "label"})
      .data_param("train.txt", 128, {784, 1})
      .Build();
  net->AddLayer(data_layer_param);
  // Dense layer
  LayerParameter dense_layer_param = LayerParameterBuilder().type("Dense")
      .bottom({"data"})
      .top({"dense"})
      .dense_param(10, true, weight_filler, bias_filler, 1)
      .Build();
  net->AddLayer(dense_layer_param);
  // Softmax cross entropy loss
  LayerParameter softmax_loss_param = LayerParameterBuilder()
      .type('SoftmaxCrossEntropyLoss')
      .bottom({"dense", "label"})
      .top({"loss"})
      .softmax_param(1)
      .Build();
  net->AddLayer(softmax_loss_param);
  return net;
}

shared_ptr<Solver<float>> GetSolver() {
  shared_ptr<Net<float>> net = BuildNet();
  SolverParameter solver_param = SolverParamBuilder().type("sgd")
      .base_lr(0.5).max_iter(1000).iter_size(1)
      .regularization_type("L2").Build();
  shared_ptr<Solver<float>> solver(
      new SGDSolver<float>(solver_param, net.get()));
  return solver;
}

int main() {
  shared_ptr<Solver<float>> solver = GetSolver();
  solver->Solve();
  return 0;
}