#ifndef TENSORFLOW_CC_EXAMPLES_MCTS_H_
#define TENSORFLOW_CC_EXAMPLES_MCTS_H_

#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "tensorflow/cc/examples/mcts_node.h"
#include "tensorflow/cc/examples/snake.h"
#include "tensorflow/core/platform/default/logging.h"

namespace snake {

double random_rollout(const SnakeMctsAdapter& state) {
  auto s = state;
  std::random_device rd;
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(Direction_MIN, Direction_MAX);

  while (!s.is_terminal()) {
    Direction candidate_d = static_cast<Direction>(distrib(gen));
    if (s.valid_action(candidate_d)) {
      s.execute(candidate_d);
    }
  }
  return s.value();
}

class Mcts {
 public:
  static Direction Strategy(const SnakeBoard16::PlayerView& p) {
    Mcts mcts(random_rollout);
    return mcts.Search(SnakeMctsAdapter(SnakeBoard16(p)));
  }
  Mcts(std::function<double(const SnakeMctsAdapter&)> rollout)
      : rollout_(rollout) {}
  Direction Search(SnakeMctsAdapter state) {
    root_ = std::make_unique<Node>(state);

    for (int i = 0; i < 1000; ++i) {
      execute_round();
    }

    auto* best_child = get_best_child(root_.get(), 0.0);
    CHECK(best_child != nullptr);
    return best_child->action_;
  }

  void backpropogate(Node* node, double reward) {
    while (node != nullptr) {
      node->num_visits_++;
      node->total_reward_ += reward;
      node = node->parent_;
    }
  }

  void execute_round() {
    Node* node = select_node(root_.get());
    double reward = rollout_(node->state_);
    backpropogate(node, reward);
  }

  Node* select_node(Node* node) {
    while (!node->is_terminal_) {
      if (node->is_fully_expanded()) {
        node = get_best_child(node, exploration_constant_);
      } else {
        return node->expand();
      }
    }
    return node;
  }

  Node* get_best_child(Node* node, double exploration_value) const {
    return *std::max_element(node->children_.begin(), node->children_.end(),
                             [exploration_value](Node* a, Node* b) {
                               if (a == nullptr && b == nullptr) {
                                 return false;
                               }
                               if (a == nullptr || b == nullptr) {
                                 return a == nullptr;
                               }
                               return a->ucb(exploration_value) <
                                      b->ucb(exploration_value);
                             });
  }

 private:
  float exploration_constant_ = 2.0;
  std::function<double(const SnakeMctsAdapter&)> rollout_;
  std::unique_ptr<Node> root_;
};

}  // namespace snake

#endif