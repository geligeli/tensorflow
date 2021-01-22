#include "tensorflow/cc/examples/az_mcts.h"

#include "tensorflow/core/platform/test.h"

namespace snake {

TEST(Mcts, SavedModel) {
  Network n;

  n.Load("/home/geli/tmp/saved_model");
}

}  // namespace snake