#ifndef TENSORFLOW_CC_EXAMPLES_NETWORK_H_
#define TENSORFLOW_CC_EXAMPLES_NETWORK_H_

#include <vector>

#include "tensorflow/cc/examples/snake.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/platform/default/logging.h"

namespace snake {

class Network {
 public:
  struct Prediction {
    std::vector<double> policy;
    double value;
  };
  static tensorflow::Tensor MakeImages(
      const std::vector<SnakeMctsAdapter>& states) {
    std::vector<const SnakeMctsAdapter*> ptr_states;
    ptr_states.reserve(states.size());
    std::transform(states.begin(), states.end(), std::back_inserter(ptr_states),
                   [](const SnakeMctsAdapter& a) { return &a; });
    // for (int i = 0; i < states.size(); ++i) {
    //   ptr_states.push_back(&states[i]);
    // }
    return MakeImages(ptr_states);
  }

  static tensorflow::Tensor MakeImages(
      std::vector<const SnakeMctsAdapter*> states) {
    using namespace tensorflow;
    Tensor input(DT_FLOAT,
                 TensorShape{{static_cast<int>(states.size()), 16, 16, 3}});
    input.tensor<float, 4>().setZero();
    for (int i = 0; i < states.size(); ++i) {
      const auto apple_pos = states[i]->board().apple_position();
      input.tensor<float, 4>()(i, apple_pos.x, apple_pos.y, 2) = 1.0;
      int j = 0;
      for (const auto p : states[i]->board().p1_view().player.points()) {
        input.tensor<float, 4>()(i, p.x, p.y, 2) = ++j;
      }
      j = 0;
      for (const auto p : states[i]->board().p1_view().opponent.points()) {
        input.tensor<float, 4>()(i, p.x, p.y, 2) = ++j;
      }
    }
    return input;
  }

  void BatchPredict(std::vector<const SnakeMctsAdapter*> states,
                    std::vector<Prediction>& predictions) const {
    using namespace tensorflow;
    Tensor input = Network::MakeImages(states);
    std::vector<Tensor> output;
    predictions.resize(states.size());
    TF_CHECK_OK(model_bundle_.session->Run(
        {{"serving_default_board:0", input}},
        {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"}, {},
        &output));
    CHECK_EQ(output.size(), 2);
    for (int i = 0; i < states.size(); ++i) {
      predictions[i].policy.resize(4);
      for (int j = 0; j < 4; ++j) {
        predictions[i].policy[j] = output[0].matrix<float>()(i, j);
      }
      predictions[i].value = output[1].matrix<float>()(i, 0);
    }
    ++num_pred_calls_;
    num_pred_lines_ += states.size();
    LOG_EVERY_N_SEC(INFO, 10) << num_pred_lines_;
  }

  void Warmup() const {
    SnakeBoard16 board;
    SnakeMctsAdapter adapter(board);
    std::vector<Prediction> unused;
    BatchPredict({&adapter}, unused);
    num_pred_calls_--;
    num_pred_lines_--;
  }

  void Load(const std::string& dirname) {
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    // (*session_options.config.mutable_device_count())["GPU"] = 0;
    // session_options.config.set_log_device_placement(true);
    session_options.config.mutable_gpu_options()->set_allow_growth(false);
    session_options.config.mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(0.5);
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    TF_CHECK_OK(tensorflow::LoadSavedModel(
        session_options, run_options, dirname,
        {tensorflow::kSavedModelTagServe}, &model_bundle_));
    Warmup();
  }

  int num_pred_calls() const { return num_pred_calls_; };
  int num_pred_lines() const { return num_pred_lines_; };

 private:
  mutable std::atomic<int> num_pred_calls_{};
  mutable std::atomic<int> num_pred_lines_{};
  tensorflow::SavedModelBundle model_bundle_;
};

}  // namespace snake

#endif