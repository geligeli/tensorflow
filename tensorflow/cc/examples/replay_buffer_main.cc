#include <grpcpp/grpcpp.h>

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/examples/replay_buffer.h"
#include "tensorflow/cc/examples/snake.grpc.pb.h"
#include "tensorflow/cc/examples/snake.pb.h"
#include "tensorflow/core/platform/default/logging.h"

ABSL_FLAG(int, port, 8000, "Port for the server to listen on.");

using namespace snake;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

void RunServer() {
  std::string server_address(
      absl::StrCat("0.0.0.0:", absl::GetFlag(FLAGS_port)));
  ReplayBufferImpl service;

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  RunServer();
  return 0;
}