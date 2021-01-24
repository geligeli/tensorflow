#include "tensorflow/cc/examples/az_mcts.h"

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/test.h"

namespace snake {

TEST(Mcts, SavedModel) {
  Network n;
  n.Load("/home/geli/tmp/saved_model");
}

TEST(Mcts, BatchPredict) {
  Network n;
  n.Load("/tmp/saved_model");
  Snake p1({
      Point{0, 3},
      Point{0, 2},
      Point{0, 1},
      Point{0, 0},
  });

  Snake p2({
      Point{1, 5},
      Point{1, 4},
      Point{1, 3},
      Point{1, 2},
  });

  SnakeBoard16 state(p1, p2, [](const SnakeBoard16&) { return Point{15, 15}; });
  state.print();
  SnakeMctsAdapter adapter(state);

  std::vector<Network::Prediction> preds;
  n.BatchPredict({&adapter}, preds);
  for (const auto& p : preds) {
    LOG(ERROR) << absl::StrJoin(p.policy, ",") << " " << p.value;
  }

  n.BatchPredict({&adapter, &adapter, &adapter, &adapter, &adapter}, preds);
  for (const auto& p : preds) {
    LOG(ERROR) << absl::StrJoin(p.policy, ",") << " " << p.value;
  }
}

TEST(Mcts, ParallelMcts) {}

}  // namespace snake