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
    if (move_queued_) {
      return 0;
    }

    switch (state_.game_state()) {
      case GameState::P1_WIN:
        return 1;
      case GameState::P2_WIN:
        return -1;
      default:
        return 0;
    }
  }

  std::bitset<Direction_ARRAYSIZE> valid_actions() const {
    std::bitset<Direction_ARRAYSIZE> result;
    for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
      result.set(i, valid_action(static_cast<Direction>(i)));
    }
    return result;
  }

  bool valid_action(Direction d) const {
    if (p1_queued_move_) {
      return state_.p2_view().valid_move(d);
    }
    return state_.p1_view().valid_move(d);
  }

  bool is_terminal() const { return state_.is_terminal(); }

 private:
  SnakeBoard16 state_;
  Direction p1_queued_move_;
  bool move_queued_ = false;
};

class Node {
 public:
  explicit Node(SnakeMctsAdapter state, Node* parent)
      : state_(state),
        parent_(parent),
        is_terminal_(state_.is_terminal()),
        valid_actions_(state_.valid_actions()) {}

  SnakeMctsAdapter state_;
  Node* parent_;
  Direction action_;
  bool is_terminal_;
  bool is_fully_expanded() const {
    return num_children_expanded == valid_actions_.count();
  }
  int num_visits_ = 0;
  double total_reward_ = 0;
  double ucb(double exploration_value) const {
    return total_reward_ / num_visits_ /* *current_player */ +
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
      children_[i] = new Node(clone_state, this);
      ++num_children_expanded;
      return children_[i];
    }
    return nullptr;
  }
  const std::bitset<Direction_ARRAYSIZE> valid_actions_;
  int num_children_expanded = 0;
  std::array<Node*, Direction_ARRAYSIZE> children_;
};

double random_rollout(const SnakeMctsAdapter& state) {
  auto s = state;
  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
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
    root_ = new Node(state, nullptr);

    for (int i = 0; i < 100; ++i) {
      execute_round();
    }

    auto* best_child = get_best_child(root_, 0.0);
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
    Node* node = select_node(root_);
    double reward = rollout_(node->state_);
    backpropogate(node, reward);
  }

  Node* select_node(Node* node) {
    while (!node->is_terminal_) {
      if (node->is_fully_expanded()) {
        node = get_best_child(node, exploration_constant_);
      } else {
        return expand(node);
      }
    }
    return node;
  }

  Node* expand(Node* node) {
    return nullptr;
    /*  actions = node.state.getPossibleActions()
     for action in actions:
         if action not in node.children:
             newNode = treeNode(node.state.takeAction(action), node)
             node.children[action] = newNode
             if len(actions) == len(node.children):
                 node.isFullyExpanded = True
             return newNode

     raise Exception("Should never reach here") */
  }

  Node* get_best_child(Node* node, double exploration_value) {
    return *std::max_element(node->children_.begin(), node->children_.end(),
                             [exploration_value](Node* a, Node* b) {
                               return a->ucb(exploration_value) <
                                      b->ucb(exploration_value);
                             });
    // bestValue = float("-inf")
    /* bestNodes = []
    for child in node.children.values():
        nodeValue = node.state.getCurrentPlayer() * child.totalReward /
child.numVisits + explorationValue * math.sqrt( 2 * math.log(node.numVisits) /
child.numVisits)
    if nodeValue > bestValue: bestValue = nodeValue bestNodes =
[child] elif nodeValue == bestValue: bestNodes.append(child) return
random.choice(bestNodes) */
  }

 private:
  float exploration_constant_ = 2.0;
  std::function<double(const SnakeMctsAdapter&)> rollout_;
  Node* root_;
};
/*
def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " +
str(state)) state = state.takeAction(action) return state.getReward()
*/

}  // namespace snake

#endif