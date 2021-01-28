#ifndef TENSORFLOW_CC_EXAMPLES_SELF_PLAY_H_
#define TENSORFLOW_CC_EXAMPLES_SELF_PLAY_H_

#include "tensorflow/cc/examples/az_mcts.h"

namespace snake {

void SelfPlay(Mcts& a, Mcts& b, int num_rounds,
              std::vector<SnakeMctsAdapter>* states) {
  SnakeBoard16 state;
  SnakeMctsAdapter mcts_adapter(state);
  states->push_back(mcts_adapter);
  while (!mcts_adapter.is_terminal()) {
    auto d = a.Search(mcts_adapter, num_rounds);
    mcts_adapter.execute(d);
    states->push_back(mcts_adapter);
    d = b.Search(mcts_adapter, num_rounds);
    mcts_adapter.execute(d);
    states->push_back(mcts_adapter);
    // mcts_adapter.print(false);
  }
  //  LOG(ERROR) << "done";
}

}  // namespace snake

#endif