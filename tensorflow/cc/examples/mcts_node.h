#ifndef TENSORFLOW_CC_EXAMPLES_MCTS_NODE_H_
#define TENSORFLOW_CC_EXAMPLES_MCTS_NODE_H_

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "tensorflow/cc/examples/snake.h"
#include "tensorflow/core/platform/default/logging.h"

namespace snake {

class Node {
 public:
  explicit Node(SnakeMctsAdapter state) : Node(state, Direction::UP, nullptr) {}

  explicit Node(SnakeMctsAdapter state, Direction action, Node* parent)
      : state_(state),
        parent_(parent),
        action_(action),
        is_terminal_(state_.is_terminal()),
        valid_actions_(state_.valid_actions()) {
    CHECK(valid_actions_.count() > 0);
  }

  ~Node() {
    for (Node* n : children_) {
      if (n) {
        delete n;
      }
    }
  }

  bool is_fully_expanded() const {
    return num_children_expanded == valid_actions_.count();
  }
  double ucb(double exploration_value) const {
    return (total_reward_ / num_visits_) * state_.player() +
           exploration_value *
               sqrt(2.0 * log(parent_->num_visits_) / num_visits_);
  }
  Node* expand() {
    for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
      if (!valid_actions_.test(i) || children_[i]) {
        continue;
      }
      auto clone_state = state_;
      clone_state.execute(static_cast<Direction>(i));
      children_[i] = new Node(clone_state, static_cast<Direction>(i), this);
      ++num_children_expanded;
      return children_[i];
    }
    state_.print();
    CHECK(false) << "should never happen " << num_children_expanded;
    return nullptr;
  }

  const SnakeMctsAdapter state_;
  Node* const parent_;
  const Direction action_;
  const bool is_terminal_;
  const std::bitset<Direction_ARRAYSIZE> valid_actions_;

  int num_visits_ = 0;
  double total_reward_ = 0;
  size_t num_children_expanded = 0;
  std::array<Node*, Direction_ARRAYSIZE> children_ = {};
  double prior_ = 0.0;
};

}  // namespace snake

#endif