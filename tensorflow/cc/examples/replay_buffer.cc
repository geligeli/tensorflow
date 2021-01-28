
#include "tensorflow/cc/examples/replay_buffer.h"

#include "absl/flags/flag.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/logging.h"

ABSL_FLAG(int, replay_buffer_size, 256000, "Size of the replay buffer.");

using tensorflow::DT_FLOAT;
using tensorflow::TensorShape;

namespace snake {

ReplayBufferImpl::ReplayBufferImpl()
    : size_(absl::GetFlag(FLAGS_replay_buffer_size)),
      boards_(DT_FLOAT, tensorflow::TensorShape{{size_, 16, 16, 3}}),
      policies_(DT_FLOAT, tensorflow::TensorShape{{size_, 4}}),
      values_(DT_FLOAT, tensorflow::TensorShape{{size_, 1}}) {}

Status ReplayBufferImpl::StoreBuffer(ServerContext* context,
                                     const StoreBufferRequest* request,
                                     StoreBufferResponse* reply) {
  tensorflow::Tensor boards, values, policies;
  CHECK(boards.FromProto(request->game_input()));
  CHECK(values.FromProto(request->value_label()));
  CHECK(policies.FromProto(request->policy_label()));
  if (boards.shape().dim_size(0) != values.shape().dim_size(0) ||
      boards.shape().dim_size(0) != policies.shape().dim_size(0)) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "size mismatch"};
  }
  return Status::OK;
}

}  // namespace snake