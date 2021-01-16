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

#include "tensorflow/cc/examples/snake.h"
#include "tensorflow/core/platform/default/logging.h"

namespace snake {

class SnakeMctsAdapter {
 public:
  explicit SnakeMctsAdapter(SnakeBoard16 state) : state_(state) {}

  void execute(Direction d) {
    if (move_queued_) {
      state_.move(p1_queued_move_, d);
      move_queued_ = false;
    } else {
      move_queued_ = true;
      p1_queued_move_ = d;
    }
  }

  double value() const {
    switch (state_.game_state()) {
      case GameState::P1_WIN:
        return 1;
      case GameState::P2_WIN:
        return -1;
      case GameState::DRAW:
        return 0;
      default:
        break;
    }
    CHECK(false);
    return 0;
  }

  std::bitset<Direction_ARRAYSIZE> valid_actions() const {
    std::bitset<Direction_ARRAYSIZE> result;
    for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
      result.set(i, valid_action(static_cast<Direction>(i)));
    }
    CHECK(result.count() > 0);
    return result;
  }

  bool valid_action(Direction d) const {
    if (move_queued_) {
      return state_.p2_view().valid_move(d);
    }
    return state_.p1_view().valid_move(d);
  }

  bool is_terminal() const {
    return state_.is_terminal();  // || (valid_actions().count() == 0);
  }

  int player() const { return move_queued_ ? 1 : -1; }

  void print() const {
    state_.print();
    if (move_queued_) {
      std::cout << "p1 queue move: " << Direction_Name(p1_queued_move_)
                << std::endl;
    } else {
      std::cout << "no move queued for p1" << std::endl;
    }
  }

 private:
  SnakeBoard16 state_;
  Direction p1_queued_move_;
  bool move_queued_ = false;
};

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

  const SnakeMctsAdapter state_;
  Node* const parent_;
  const Direction action_;
  bool is_terminal_;
  bool is_fully_expanded() const {
    return num_children_expanded == valid_actions_.count();
  }
  int num_visits_ = 0;
  double total_reward_ = 0;
  double ucb(double exploration_value) const {
    // if (parent_->parent_ == nullptr && exploration_value == 0.0)
    //   LOG(ERROR) << Direction_Name(action_) << " " << total_reward_ << " "
    //              << num_visits_ << " " << state_.player();
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
      // if (parent_ == nullptr)
      //   LOG(ERROR) << "expanding " << i << " "
      //              << Direction_Name(static_cast<Direction>(i));
      clone_state.execute(static_cast<Direction>(i));
      children_[i] = new Node(clone_state, static_cast<Direction>(i), this);
      ++num_children_expanded;
      return children_[i];
    }
    state_.print();
    CHECK(false) << "should never happen " << num_children_expanded;
    return nullptr;
  }
  const std::bitset<Direction_ARRAYSIZE> valid_actions_;
  size_t num_children_expanded = 0;
  std::array<Node*, Direction_ARRAYSIZE> children_ = {};
  ~Node() {
    for (Node* n : children_) {
      if (n) {
        delete n;
      }
    }
  }
};

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
  Mcts(std::function<double(const SnakeMctsAdapter&)> rollout)
      : rollout_(rollout) {}
  Direction Search(SnakeMctsAdapter state) {
    root_ = std::make_unique<Node>(state);

    for (int i = 0; i < 10000; ++i) {
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