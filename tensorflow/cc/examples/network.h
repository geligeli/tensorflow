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
  void BatchPredict(std::vector<const SnakeMctsAdapter*> states,
                    std::vector<Prediction>& predictions) const {
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
  }
  void Load(const std::string& dirname) {
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    // (*session_options.config.mutable_device_count())["GPU"] = 0;
    // session_options.config.set_log_device_placement(true);
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    TF_CHECK_OK(tensorflow::LoadSavedModel(
        session_options, run_options, dirname,
        {tensorflow::kSavedModelTagServe}, &model_bundle_));
  }

 private:
  tensorflow::SavedModelBundle model_bundle_;
};

}  // namespace snake

#endif