#ifndef LICH_SRC_FILLER_H_
#define LICH_SRC_FILLER_H_

#include "lich/proto/filler_param.pb.h"
#include "lich/src/tensor.h"
#include "lich/lib/math.h"

#include "glog/logging.h"

namespace lich {

template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& filler_param) 
      : filler_param_(filler_param) {}
  virtual ~Filler() {}
  virtual void Fill(Tensor<Dtype>* tensor) = 0;
  
  static Filler* GetFiller(const FillerParameter& filler_param);
 protected:
  FillerParameter filler_param_;
};

template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& filler_param)
      : Filler<Dtype>(filler_param) {}
  virtual void Fill(Tensor<Dtype>* tensor) override {
    Dtype* data = tensor->mutable_data();
    const int count = tensor->count();
    const Dtype value = static_cast<Dtype>(this->filler_param_.value());
    lich_set(count, value, data);
  }
};

template <typename Dtype>
class GuassianFiller : public Filler<Dtype> {
 public:
  explicit GuassianFiller(const FillerParameter& filler_param)
      : Filler<Dtype>(filler_param) {}
  virtual void Fill(Tensor<Dtype>* tensor) override {
    Dtype* data = tensor->mutable_data();
    const int count = tensor->count();
    Dtype mean = static_cast<Dtype>(this->filler_param_.mean());
    Dtype std = static_cast<Dtype>(this->filler_param_.std());
    lich_rng_gaussian(count, mean, std, data);
  }
};

template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& filler_param)
      : Filler<Dtype>(filler_param) {}
  virtual void Fill(Tensor<Dtype>* tensor) override {
    Dtype* data = tensor->mutable_data();
    const int count = tensor->count();
    Dtype min = static_cast<Dtype>(this->filler_param_.min());
    Dtype max = static_cast<Dtype>(this->filler_param_.max());
    lich_rng_uniform(count, min, max, data);
  }
};

template <typename Dtype>
Filler<Dtype>* Filler<Dtype>::GetFiller(const FillerParameter& filler_param) {
  string type = filler_param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(filler_param);
  }
  else if (type == "guassian") {
    return new GuassianFiller<Dtype>(filler_param);
  }
  else if (type == "uniform") {
    return new UniformFiller<Dtype>(filler_param);
  }
  else {
    CHECK(false) << "Filler type '" << type << "' not exisit";
  }
  // Just to remove compile warning.
  return nullptr;
}

} // namespace lich 

#endif