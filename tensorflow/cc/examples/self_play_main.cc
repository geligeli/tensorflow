#include <grpcpp/grpcpp.h>

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/examples/network_fiber_batch.h"
#include "tensorflow/cc/examples/self_play.h"
#include "tensorflow/cc/examples/snake.grpc.pb.h"
#include "tensorflow/cc/examples/snake.pb.h"
#include "tensorflow/core/platform/default/logging.h"

ABSL_FLAG(std::string, replay_buffer, "localhost:8000",
          "gRPC endpoint of replay buffer.");

using namespace snake;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

// void RunServer() {
//   std::string server_address(
//       absl::StrCat("0.0.0.0:", absl::GetFlag(FLAGS_port)));
//   ReplayBufferImpl service;

//   ServerBuilder builder;
//   // Listen on the given address without any authentication mechanism.
//   builder.AddListeningPort(server_address,
//   grpc::InsecureServerCredentials());
//   // Register "service" as the instance through which we'll communicate with
//   // clients. In this case it corresponds to an *synchronous* service.
//   builder.RegisterService(&service);
//   // Finally assemble the server.
//   std::unique_ptr<Server> server(builder.BuildAndStart());
//   LOG(INFO) << "Server listening on " << server_address;

//   // Wait for the server to shutdown. Note that some other thread must be
//   // responsible for shutting down the server for this call to ever return.
//   server->Wait();
// }
void SelfPlay(snake::ReplayBuffer::Stub& stub) {
  using namespace snake;
  Network n;
  n.Load("/tmp/saved_model");

  auto self_play_loop = [&]() {
    NetworkFiberBatch batcher_a(n);
    NetworkFiberBatch batcher_b(n);
    int num_games = 1024;
    boost::fibers::fiber fib1, fib2;
    {
      std::vector<boost::fibers::fiber> game_fibers;
      for (int i = 0; i < num_games; ++i) {
        game_fibers.emplace_back([a = Mcts(batcher_a.new_fiber()),
                                  b = Mcts(batcher_b.new_fiber()), i = i,
                                  &stub]() mutable {
          while (true) {
            std::vector<SnakeMctsAdapter> states;
            SelfPlay(a, b, 50, &states);

            auto tensor = Network::MakeImages(states);
            StoreBufferRequest request;
            tensor.AsProtoTensorContent(request.mutable_game_input());
            StoreBufferResponse response;

            // Here we can use the stub's newly available method we just
            // added.
            grpc::ClientContext context;
            Status status = stub.StoreBuffer(&context, request, &response);
            if (!status.ok()) {
              LOG(ERROR) << status.error_code() << ": "
                         << status.error_message();
            }

            LOG(ERROR) << "game " << i << " done in n=" << states.size()
                       << " moves";
          }
        });
      };
      fib1 = batcher_a.start_prediction_fiber();
      fib2 = batcher_b.start_prediction_fiber();
      std::for_each(game_fibers.begin(), game_fibers.end(),
                    std::mem_fn(&boost::fibers::fiber::join));
    }
    fib1.join();
    fib2.join();
  };
  std::vector<std::thread> threads(8);
  std::map<std::thread::id, std::string> thread_names;
  int i = 0;
  for (auto& t : threads) {
    t = std::thread(self_play_loop);
    thread_names[t.get_id()] = absl::StrCat(++i);
  }
  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto channel = grpc::CreateChannel(absl::GetFlag(FLAGS_replay_buffer),
                                     grpc::InsecureChannelCredentials());
  snake::ReplayBuffer::Stub stub(channel);

  SelfPlay(stub);
  return 0;
}
