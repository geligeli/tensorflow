#ifndef TENSORFLOW_CC_EXAMPLES_AZ_MCTS_H_
#define TENSORFLOW_CC_EXAMPLES_AZ_MCTS_H_

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
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
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

  /*
  def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
      node = root
      scratch_game = game.clone()
      search_path = [node]

      while node.expanded():
        action, node = select_child(config, node)
        scratch_game.apply(action)
        search_path.append(node)

      value = evaluate(node, scratch_game, network)
      backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root
  */
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

class Network {
 public:
  struct Prediction {
    std::vector<double> policy;
    double value;
  };
  void BatchPredict(std::vector<const SnakeMctsAdapter*> states,
                    std::vector<Prediction>& predictions) const {
    using namespace tensorflow;
    Tensor input(DT_FLOAT,
                 TensorShape{{static_cast<int>(states.size()), 16, 16, 3}});
    for (int i = 0; i < input.NumElements(); ++i) {
      input.flat<float>()(i) = 0;
    }
    std::vector<Tensor> output;
    predictions.resize(states.size());
    TF_CHECK_OK(model_bundle_.session->Run(
        {{"serving_default_board:0", input}},
        {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"}, {},
        &output));
    CHECK_EQ(output.size(), 2);
    for (int i = 0; i < states.size(); ++i) {
      for (int j = 0; j < 4; ++j) {
        predictions[i].policy[j] = output[0].matrix<float>()(i, j);
      }
      predictions[i].value = output[1].scalar<float>()();
    }
  }
  void Load(const std::string& dirname) {
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    // (*session_options.config.mutable_device_count())["GPU"] = 0;
    session_options.config.set_log_device_placement(true);
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    TF_CHECK_OK(tensorflow::LoadSavedModel(
        session_options, run_options, dirname,
        {tensorflow::kSavedModelTagServe}, &model_bundle_));
  }

 private:
  tensorflow::SavedModelBundle model_bundle_;
};

/*
"""Pseudocode description of the AlphaZero algorithm."""

from __future__ import google_type_annotations
from __future__ import division

import math
import numpy
import tensorflow as tf
from typing import List

##########################
####### Helpers ##########


class AlphaZeroConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for
shogi. self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }


class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class Game(object):

  def __init__(self, history=None):
    self.history = history or []
    self.child_visits = []
    self.num_actions = 4672  # action space size for chess; 11259 for shogi, 362
for Go

  def terminal(self):
    # Game specific termination rules.
    pass

  def terminal_value(self, to_play):
    # Game specific value.
    pass

  def legal_actions(self):
    # Game specific calculation of legal actions.
    return []

  def clone(self):
    return Game(list(self.history))

  def apply(self, action):
    self.history.append(action)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.itervalues())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int):
    return (self.terminal_value(state_index % 2),
            self.child_visits[state_index])

  def to_play(self):
    return len(self.history) % 2


class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    move_sum = float(sum(len(g.history) for g in self.buffer))
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class Network(object):

  def inference(self, image):
    return (-1, {})  # Value, Policy

  def get_weights(self):
    # Returns the weights of this network.
    return []


class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.iterkeys())]
    else:
      return make_uniform_network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for i in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.iteritems()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.iteritems())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  value, policy_logits = network.inference(game.make_image(-1))

  # Expand the node.
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.itervalues())
  for action, p in policy.iteritems():
    node.children[action] = Node(p / policy_sum)
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                         config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (
        tf.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=target_policy))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()

*/

}  // namespace snake

#endif