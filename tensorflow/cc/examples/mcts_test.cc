#include "tensorflow/cc/examples/mcts.h"

#include <boost/coroutine2/coroutine.hpp>
#include <boost/fiber/all.hpp>
#include <chrono>
#include <list>
#include <thread>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
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
  EXPECT_EQ(d, Direction::DOWN);
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

  SnakeBoard16 state(p1, p2, [](const SnakeBoard16&) { return Point{15, 15}; });
  state.print();
  SnakeMctsAdapter adapter(state);
  Mcts mcts(random_rollout);
  auto d = mcts.Search(adapter);
  EXPECT_EQ(d, Direction::DOWN);
  adapter.execute(d);
  d = mcts.Search(adapter);
  EXPECT_EQ(d, Direction::LEFT);
  adapter.execute(d);
  //   adapter.print();
}

TEST(Mcts, MctsMultipleMoves) {
  SnakeBoard16 state;
  SnakeMctsAdapter adapter(state);
  while (!adapter.is_terminal()) {
    Mcts mcts(random_rollout);
    auto d = mcts.Search(adapter);
    adapter.execute(d);
    d = mcts.Search(adapter);
    adapter.execute(d);
    adapter.print();
  }
  adapter.print();
}

TEST(Mcts, TestMctsStrategy) { RunGame(GreedyStrategy, Mcts::Strategy); }

TEST(Mcts, SavedModel) {
  using namespace tensorflow;
  // Load
  SavedModelBundle model_bundle;
  SessionOptions session_options = SessionOptions();
  // (*session_options.config.mutable_device_count())["GPU"] = 0;
  session_options.config.set_log_device_placement(true);
  RunOptions run_options = RunOptions();
  TF_CHECK_OK(LoadSavedModel(session_options, run_options,
                             "/home/geli/tmp/saved_model",
                             {kSavedModelTagServe}, &model_bundle));

  // LOG(ERROR) << model_bundle.meta_graph_def.DebugString();
  for (const auto& s : model_bundle.GetSignatures()) {
    LOG(ERROR) << s.first;
    LOG(ERROR) << s.second.DebugString();
  }

  Tensor input(DT_FLOAT, TensorShape{{4096, 16, 16, 3}});
  for (int i = 0; i < input.NumElements(); ++i) {
    input.flat<float>()(i) = 0;
  }
  std::vector<Tensor> output;
  TF_CHECK_OK(model_bundle.session->Run({{"serving_default_board:0", input}},
                                        {"StatefulPartitionedCall:0"}, {},
                                        &output));
  ASSERT_EQ(output.size(), 1);

  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) {
    std::vector<Tensor> output;
    TF_CHECK_OK(model_bundle.session->Run({{"serving_default_board:0", input}},
                                          {"StatefulPartitionedCall:0"}, {},
                                          &output));
    ASSERT_EQ(output.size(), 1);
    LOG(ERROR) << output[0].shape().DebugString();
    LOG(ERROR) << output[0].DebugString(8);
  }
  std::cout << "Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - begin)
                   .count()
            << "[µs]" << std::endl;
  // model_bundle.session->Run()
  // bundle.session->Run({}, {"filename_tensor:0"}, {}, &path_outputs));
  //   Tensor

  // Status runStatus = model_bundle.GetSession()->Run(
  //     {{"serving_default_input_tensor:0", input_tensor}}, outputLayer, {},
  //     &outputs);

  // Inference
  // outputLayer = {"StatefulPartitionedCall:0", "StatefulPartitionedCall:4",
  //                "StatefulPartitionedCall:1", "StatefulPartitionedCall:5",
  //                "StatefulPartitionedCall:3"};
  // Status runStatus = model_bundle.GetSession()->Run(
  //     {{"serving_default_input_tensor:0", input_tensor}}, outputLayer, {},
  //     &outputs);
}

TEST(Coroutine, Test) {
  boost::coroutines2::coroutine<int>::pull_type source(
      [](boost::coroutines2::coroutine<int>::push_type& sink) {
        int first = 1, second = 1;
        sink(first);
        sink(second);
        for (int i = 0; i < 8; ++i) {
          int third = first + second;
          first = second;
          second = third;
          sink(third);
        }
      });
  for (auto i : source) {
    std::cout << i << " ";
  }
  std::cout << "\nDone" << std::endl;
}

template <typename INPUT, typename OUTPUT>
struct Batching {
  struct Work {
    INPUT input;
    boost::fibers::promise<OUTPUT> r;
  };

  using ChannelType = boost::fibers::unbuffered_channel<Work>;

  struct Closer {
    void operator()(ChannelType* c) { c->close(); }
  };
  using Base = std::unique_ptr<ChannelType, Closer>;
  struct Foo : public Base {
    using Base::Base;
    OUTPUT BatchProcess(INPUT input) const {
      Work w;
      w.input = input;
      auto f = w.r.get_future();
      Base::get()->push(std::move(w));
      return f.get();
    }
  };
  Foo new_channel() {
    channels_.emplace_back();
    return Foo{&channels_.back(), Closer()};
  }
  typename boost::coroutines2::coroutine<Work>::pull_type get_batch() {
    return typename boost::coroutines2::coroutine<Work>::pull_type(
        [this](typename boost::coroutines2::coroutine<Work>::push_type& sink) {
          for (auto it = channels_.begin(); it != channels_.end();) {
            Work w;
            if (it->pop(w) != boost::fibers::channel_op_status::success) {
              it = channels_.erase(it);
              continue;
            } else {
              ++it;
            }
            sink(std::move(w));
          }
        });
  }

 private:
  std::list<ChannelType> channels_;
};

TEST(Fiber, Test) {
  Batching<int, int> c;

  auto batch_compute = [&c]() {
    while (true) {
      auto batch = c.get_batch();
      std::vector<boost::fibers::promise<int>> results;
      int result = 0;
      int batch_size = 0;
      for (auto&& w : batch) {
        ++batch_size;
        result += w.input;
        results.push_back(std::move(w.r));
      }
      if (batch_size == 0) {
        break;
      }
      for (auto& r : results) {
        r.set_value(result);
      }
    }
  };

  auto leaf_comp = [](const Batching<int, int>::Foo& f, int i) {
    for (int j = 0; j < i; ++j) {
      auto r = f.BatchProcess(j);
      std::cout << i << ": f(" << j << ")=" << r << std::endl;
    }
  };

  boost::fibers::fiber l1{[leaf_comp, f = std::move(c.new_channel()), i = 3]() {
    leaf_comp(f, i);
  }};
  boost::fibers::fiber l2{[leaf_comp, f = std::move(c.new_channel()), i = 4]() {
    leaf_comp(f, i);
  }};
  boost::fibers::fiber l3{[leaf_comp, f = std::move(c.new_channel()), i = 5]() {
    leaf_comp(f, i);
  }};
  boost::fibers::fiber bc{batch_compute};

  l1.join();
  std::cout << "l1 joined" << std::endl;
  l2.join();
  std::cout << "l2 joined" << std::endl;
  l3.join();
  std::cout << "l3 joined" << std::endl;
  bc.join();
}

}  // namespace snake