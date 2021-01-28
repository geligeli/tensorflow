#ifndef TENSORFLOW_CC_EXAMPLES_NETWORK_FIBER_BATCH_H_
#define TENSORFLOW_CC_EXAMPLES_NETWORK_FIBER_BATCH_H_

#include "tensorflow/cc/examples/mcts_prediction_batching.h"
#include "tensorflow/cc/examples/network.h"

namespace snake {

class NetworkFiberBatch {
 public:
  explicit NetworkFiberBatch(const Network& network) : network_(network) {}

  NetworkFiberBatch(const NetworkFiberBatch&) = delete;
  NetworkFiberBatch(NetworkFiberBatch&&) = delete;
  NetworkFiberBatch& operator=(const NetworkFiberBatch&) = delete;
  NetworkFiberBatch& operator=(NetworkFiberBatch&&) = delete;

  using Batcher = Batching<const SnakeMctsAdapter*, Network::Prediction>;
  using BatchFiber = Batcher::BatchFiber;

  BatchFiber new_fiber() {
    CHECK(!prediction_fiber_started_)
        << "Can't invoke new_fiber after call to start_prediction_fiber";
    return batcher_.new_fiber();
  }

  boost::fibers::fiber start_prediction_fiber() {
    prediction_fiber_started_ = true;
    return boost::fibers::fiber(
        std::bind(&NetworkFiberBatch::PredictionLoop, this));
  }

 private:
  void PredictionLoop() {
    int i = 0;
    while (true) {
      ++i;
      auto batch = batcher_.get_batch();
      std::vector<boost::fibers::promise<Network::Prediction>> results;
      std::vector<const SnakeMctsAdapter*> input;
      // int batch_num = 0;
      for (auto&& w : batch) {
        // LOG(ERROR) << batch_num++;
        input.push_back(w.input);
        results.push_back(std::move(w.r));
      }

      // if (i % 10 == 0) {
      //   LOG(INFO) << "[" << std::this_thread::get_id() << "]: " << i
      //             << " batch prediction calls, current batch size="
      //             << input.size();
      // }
      if (input.empty()) {
        break;
      }
      // LOG(ERROR) << input.size();
      std::vector<Network::Prediction> output;
      network_.BatchPredict(input, output);
      CHECK_EQ(output.size(), results.size());
      for (int i = 0; i < output.size(); ++i) {
        results[i].set_value(std::move(output[i]));
      }
      // LOG(ERROR) << "done" << output.size();
    }
  }

  bool prediction_fiber_started_ = false;
  Batcher batcher_;
  const Network& network_;
};

}  // namespace snake

#endif