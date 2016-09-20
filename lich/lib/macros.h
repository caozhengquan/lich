#ifndef LICH_LIB_MACROS_H_
#define LICH_LIB_MACROS_H_

#include "lich/src/common.h"

namespace lich {

#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
  ClassName(const ClassName& rhs) = delete; \
  void operator=(const ClassName&) = delete;

#define INSTANTIATE_CLASS(ClassName) \
  template class ClassName<float>; \
  template class ClassName<double>

const vector<float> DEFAULT_FVECTOR;

} // namespace lich 

#endif