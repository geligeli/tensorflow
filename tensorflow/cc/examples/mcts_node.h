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

  std::string graphviz_dot() const {
    struct GraphBuilder {
      std::vector<std::string> lines;
      void operator()(const Node* node, std::string name) {
        if (node == nullptr) {
          return;
        }
        auto info = absl::StrCat("reward=", node->total_reward_, " ",
                                 "num_visits=", node->num_visits_);

        lines.push_back(absl::StrCat("  ", name,
                                     "[shape=none margin=0 label=< ",
                                     node->state_.html(info), " >];"));
        for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
          const Node* child = node->children_[i];
          if (child == nullptr) {
            continue;
          }
          auto child_name = name + Direction_Name(static_cast<Direction>(i));
          lines.push_back(absl::StrCat(
              "  ", name, "->", child_name, " [label=\"",
              Direction_Name(static_cast<Direction>(i)), " ucb=", child->ucb(0.0), "\"];"));
          this->operator()(child, child_name);
        }
      }
    };

    GraphBuilder builder;
    builder.lines.push_back("digraph G {");
    builder(this, "root");
    builder.lines.push_back("}");
    return absl::StrJoin(builder.lines, "\n");
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