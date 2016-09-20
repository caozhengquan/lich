#include "lich/src/tensor.h"
#include "lich/lib/math.h"
#include <glog/logging.h>

namespace lich {

template <typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<int>& shape) : capacity_(0) {
    Reshape(shape);
}

template <typename Dtype>
void Tensor<Dtype>::Reshape(const std::vector<int>& shape) {
    count_ = 1;
    shape_.resize(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
        CHECK_GE(shape[i], 0);
        shape_[i] = shape[i];
        count_ *= shape[i];
    }
    if (count_ > capacity_) {
        capacity_ = count_;
        // TODO(wzpfish): Delay Allocating the memory.
        data_.reset(new Dtype[sizeof(Dtype) * capacity_]);
        diff_.reset(new Dtype[sizeof(Dtype) * capacity_]);
    }
}

template <typename Dtype>
void Tensor<Dtype>::Update() {
    lich_axpy(count_, static_cast<Dtype>(-1), diff(), mutable_data());
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::data() const {
    return reinterpret_cast<Dtype*>(data_.get());
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::diff() const {
    return reinterpret_cast<Dtype*>(diff_.get());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_data() {
    return const_cast<Dtype*>(static_cast<const Tensor&>(*this).data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_diff() {
    return const_cast<Dtype*>(static_cast<const Tensor&>(*this).diff());
}

INSTANTIATE_CLASS(Tensor);

} // namespace lich 