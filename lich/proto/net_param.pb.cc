// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: lich/proto/net_param.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "lich/proto/net_param.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace lich {

namespace {

const ::google::protobuf::Descriptor* NetParameter_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  NetParameter_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_lich_2fproto_2fnet_5fparam_2eproto() {
  protobuf_AddDesc_lich_2fproto_2fnet_5fparam_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "lich/proto/net_param.proto");
  GOOGLE_CHECK(file != NULL);
  NetParameter_descriptor_ = file->message_type(0);
  static const int NetParameter_offsets_[3] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NetParameter, name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NetParameter, phase_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NetParameter, layer_),
  };
  NetParameter_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      NetParameter_descriptor_,
      NetParameter::default_instance_,
      NetParameter_offsets_,
      -1,
      -1,
      -1,
      sizeof(NetParameter),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NetParameter, _internal_metadata_),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(NetParameter, _is_default_instance_));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_lich_2fproto_2fnet_5fparam_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      NetParameter_descriptor_, &NetParameter::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_lich_2fproto_2fnet_5fparam_2eproto() {
  delete NetParameter::default_instance_;
  delete NetParameter_reflection_;
}

void protobuf_AddDesc_lich_2fproto_2fnet_5fparam_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::lich::protobuf_AddDesc_lich_2fproto_2flayer_5fparam_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\032lich/proto/net_param.proto\022\004lich\032\034lich"
    "/proto/layer_param.proto\"]\n\014NetParameter"
    "\022\014\n\004name\030\001 \001(\t\022\032\n\005phase\030\002 \001(\0162\013.lich.Pha"
    "se\022#\n\005layer\030\003 \003(\0132\024.lich.LayerParameterb"
    "\006proto3", 167);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "lich/proto/net_param.proto", &protobuf_RegisterTypes);
  NetParameter::default_instance_ = new NetParameter();
  NetParameter::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_lich_2fproto_2fnet_5fparam_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_lich_2fproto_2fnet_5fparam_2eproto {
  StaticDescriptorInitializer_lich_2fproto_2fnet_5fparam_2eproto() {
    protobuf_AddDesc_lich_2fproto_2fnet_5fparam_2eproto();
  }
} static_descriptor_initializer_lich_2fproto_2fnet_5fparam_2eproto_;

namespace {

static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD;
static void MergeFromFail(int line) {
  GOOGLE_CHECK(false) << __FILE__ << ":" << line;
}

}  // namespace


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int NetParameter::kNameFieldNumber;
const int NetParameter::kPhaseFieldNumber;
const int NetParameter::kLayerFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

NetParameter::NetParameter()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:lich.NetParameter)
}

void NetParameter::InitAsDefaultInstance() {
  _is_default_instance_ = true;
}

NetParameter::NetParameter(const NetParameter& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:lich.NetParameter)
}

void NetParameter::SharedCtor() {
    _is_default_instance_ = false;
  ::google::protobuf::internal::GetEmptyString();
  _cached_size_ = 0;
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  phase_ = 0;
}

NetParameter::~NetParameter() {
  // @@protoc_insertion_point(destructor:lich.NetParameter)
  SharedDtor();
}

void NetParameter::SharedDtor() {
  name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (this != default_instance_) {
  }
}

void NetParameter::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* NetParameter::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return NetParameter_descriptor_;
}

const NetParameter& NetParameter::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_lich_2fproto_2fnet_5fparam_2eproto();
  return *default_instance_;
}

NetParameter* NetParameter::default_instance_ = NULL;

NetParameter* NetParameter::New(::google::protobuf::Arena* arena) const {
  NetParameter* n = new NetParameter;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void NetParameter::Clear() {
// @@protoc_insertion_point(message_clear_start:lich.NetParameter)
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  phase_ = 0;
  layer_.Clear();
}

bool NetParameter::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:lich.NetParameter)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional string name = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), this->name().length(),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "lich.NetParameter.name"));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(16)) goto parse_phase;
        break;
      }

      // optional .lich.Phase phase = 2;
      case 2: {
        if (tag == 16) {
         parse_phase:
          int value;
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   int, ::google::protobuf::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
          set_phase(static_cast< ::lich::Phase >(value));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(26)) goto parse_layer;
        break;
      }

      // repeated .lich.LayerParameter layer = 3;
      case 3: {
        if (tag == 26) {
         parse_layer:
          DO_(input->IncrementRecursionDepth());
         parse_loop_layer:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtualNoRecursionDepth(
                input, add_layer()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(26)) goto parse_loop_layer;
        input->UnsafeDecrementRecursionDepth();
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:lich.NetParameter)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:lich.NetParameter)
  return false;
#undef DO_
}

void NetParameter::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:lich.NetParameter)
  // optional string name = 1;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), this->name().length(),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "lich.NetParameter.name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->name(), output);
  }

  // optional .lich.Phase phase = 2;
  if (this->phase() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteEnum(
      2, this->phase(), output);
  }

  // repeated .lich.LayerParameter layer = 3;
  for (unsigned int i = 0, n = this->layer_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      3, this->layer(i), output);
  }

  // @@protoc_insertion_point(serialize_end:lich.NetParameter)
}

::google::protobuf::uint8* NetParameter::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:lich.NetParameter)
  // optional string name = 1;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), this->name().length(),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "lich.NetParameter.name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->name(), target);
  }

  // optional .lich.Phase phase = 2;
  if (this->phase() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteEnumToArray(
      2, this->phase(), target);
  }

  // repeated .lich.LayerParameter layer = 3;
  for (unsigned int i = 0, n = this->layer_size(); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        3, this->layer(i), target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:lich.NetParameter)
  return target;
}

int NetParameter::ByteSize() const {
// @@protoc_insertion_point(message_byte_size_start:lich.NetParameter)
  int total_size = 0;

  // optional string name = 1;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->name());
  }

  // optional .lich.Phase phase = 2;
  if (this->phase() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::EnumSize(this->phase());
  }

  // repeated .lich.LayerParameter layer = 3;
  total_size += 1 * this->layer_size();
  for (int i = 0; i < this->layer_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->layer(i));
  }

  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void NetParameter::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:lich.NetParameter)
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const NetParameter* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const NetParameter>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:lich.NetParameter)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:lich.NetParameter)
    MergeFrom(*source);
  }
}

void NetParameter::MergeFrom(const NetParameter& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:lich.NetParameter)
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  layer_.MergeFrom(from.layer_);
  if (from.name().size() > 0) {

    name_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.name_);
  }
  if (from.phase() != 0) {
    set_phase(from.phase());
  }
}

void NetParameter::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:lich.NetParameter)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void NetParameter::CopyFrom(const NetParameter& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:lich.NetParameter)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool NetParameter::IsInitialized() const {

  return true;
}

void NetParameter::Swap(NetParameter* other) {
  if (other == this) return;
  InternalSwap(other);
}
void NetParameter::InternalSwap(NetParameter* other) {
  name_.Swap(&other->name_);
  std::swap(phase_, other->phase_);
  layer_.UnsafeArenaSwap(&other->layer_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata NetParameter::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = NetParameter_descriptor_;
  metadata.reflection = NetParameter_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// NetParameter

// optional string name = 1;
void NetParameter::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 const ::std::string& NetParameter::name() const {
  // @@protoc_insertion_point(field_get:lich.NetParameter.name)
  return name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 void NetParameter::set_name(const ::std::string& value) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:lich.NetParameter.name)
}
 void NetParameter::set_name(const char* value) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:lich.NetParameter.name)
}
 void NetParameter::set_name(const char* value, size_t size) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:lich.NetParameter.name)
}
 ::std::string* NetParameter::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:lich.NetParameter.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 ::std::string* NetParameter::release_name() {
  // @@protoc_insertion_point(field_release:lich.NetParameter.name)
  
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
 void NetParameter::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    
  } else {
    
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:lich.NetParameter.name)
}

// optional .lich.Phase phase = 2;
void NetParameter::clear_phase() {
  phase_ = 0;
}
 ::lich::Phase NetParameter::phase() const {
  // @@protoc_insertion_point(field_get:lich.NetParameter.phase)
  return static_cast< ::lich::Phase >(phase_);
}
 void NetParameter::set_phase(::lich::Phase value) {
  
  phase_ = value;
  // @@protoc_insertion_point(field_set:lich.NetParameter.phase)
}

// repeated .lich.LayerParameter layer = 3;
int NetParameter::layer_size() const {
  return layer_.size();
}
void NetParameter::clear_layer() {
  layer_.Clear();
}
const ::lich::LayerParameter& NetParameter::layer(int index) const {
  // @@protoc_insertion_point(field_get:lich.NetParameter.layer)
  return layer_.Get(index);
}
::lich::LayerParameter* NetParameter::mutable_layer(int index) {
  // @@protoc_insertion_point(field_mutable:lich.NetParameter.layer)
  return layer_.Mutable(index);
}
::lich::LayerParameter* NetParameter::add_layer() {
  // @@protoc_insertion_point(field_add:lich.NetParameter.layer)
  return layer_.Add();
}
::google::protobuf::RepeatedPtrField< ::lich::LayerParameter >*
NetParameter::mutable_layer() {
  // @@protoc_insertion_point(field_mutable_list:lich.NetParameter.layer)
  return &layer_;
}
const ::google::protobuf::RepeatedPtrField< ::lich::LayerParameter >&
NetParameter::layer() const {
  // @@protoc_insertion_point(field_list:lich.NetParameter.layer)
  return layer_;
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace lich

// @@protoc_insertion_point(global_scope)