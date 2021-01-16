#include "tensorflow/cc/examples/mcts.h"

#include <thread>

#include "tensorflow/core/platform/test.h"

namespace snake {

TEST(Mcts, SinglePredictionTest) {
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

  SnakeMctsAdapter adapter(state);
  adapter.print();

  Mcts mcts(random_rollout);

  auto d = mcts.Search(adapter);
  LOG(ERROR) << Direction_Name(d);
  adapter.execute(Direction::DOWN);
  adapter.print();
}

TEST(Mcts, MctsTest) {
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

  int p = 121;
  SnakeBoard16 state(p1, p2, [](const SnakeBoard16&) { return Point{15, 15}; });

  state.print();

  SnakeMctsAdapter adapter(state);
  Mcts mcts(random_rollout);

  auto d = mcts.Search(adapter);
  LOG(ERROR) << Direction_Name(d);
  adapter.execute(d);
  LOG(ERROR) << "searching second player";
  d = mcts.Search(adapter);
  LOG(ERROR) << Direction_Name(d);
  adapter.execute(d);
  adapter.print();
}

TEST(Mcts, MctsMultipleMoves) {
  /*Snake p1({
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
*/
  SnakeBoard16 state;
  // (p1, p2, [](const SnakeBoard16&) { return Point{15, 15}; });
  SnakeMctsAdapter adapter(state);

  while (!adapter.is_terminal()) {
    Mcts mcts(random_rollout);
    auto d = mcts.Search(adapter);
    adapter.execute(d);
    d = mcts.Search(adapter);
    adapter.execute(d);
    adapter.print();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  adapter.print();
}

}  // namespace snake