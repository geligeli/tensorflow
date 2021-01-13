#include "tensorflow/cc/examples/mcts.h"

#include "tensorflow/core/platform/test.h"

namespace snake {

TEST(Mcts, MctsTest) {
  Snake p1({
      Point{0, 0},
      Point{0, 1},
      Point{0, 2},
      Point{0, 3},
  });

  Snake p2({
      Point{1, 2},
      Point{1, 3},
      Point{1, 4},
      Point{1, 5},
  });

  SnakeBoard16 state(p1, p2, [](const SnakeBoard16&) { return Point{15, 15}; });

  state.print();

  SnakeMctsAdapter adapter(state);
  Mcts mcts(random_rollout);

  mcts.Search(adapter);
}

}  // namespace snake