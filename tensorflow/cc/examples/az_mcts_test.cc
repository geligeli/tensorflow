#include "tensorflow/cc/examples/az_mcts.h"

#include "absl/strings/str_join.h"
#include "tensorflow/cc/examples/network_fiber_batch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace snake {

TEST(Mcts, BatchPredict) {
  Network n;
  n.Load("/tmp/saved_model");
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

  std::vector<Network::Prediction> preds;
  n.BatchPredict({&adapter}, preds);
  for (const auto& p : preds) {
    LOG(ERROR) << absl::StrJoin(p.policy, ",") << " " << p.value;
  }

  n.BatchPredict({&adapter, &adapter, &adapter, &adapter, &adapter}, preds);
  for (const auto& p : preds) {
    LOG(ERROR) << absl::StrJoin(p.policy, ",") << " " << p.value;
  }
}

TEST(Mcts, ParallelMcts) {
  Network n;
  n.Load("/tmp/saved_model");
  NetworkFiberBatch batcher(n);

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

  boost::fibers::fiber prediction_loop_fiber;
  {
    std::vector<boost::fibers::fiber> fibers;
    std::vector<Mcts> mcts_vec;
    for (int i = 0; i < 100; ++i) {
      mcts_vec.emplace_back(batcher.new_fiber());
    }
    prediction_loop_fiber = batcher.start_prediction_fiber();
    for (auto& mcts : mcts_vec) {
      fibers.emplace_back([&m = mcts, &adapter]() {
        auto d = m.Search(adapter, 10);
        LOG(INFO) << Direction_Name(d);
      });
    }

    std::for_each(fibers.begin(), fibers.end(),
                  [](boost::fibers::fiber& f) { f.join(); });
  }
  prediction_loop_fiber.join();

  LOG(INFO) << "Num prediction batch calls " << n.num_pred_calls();
  LOG(INFO) << "Num prediction calls " << n.num_pred_lines();
}

void BM_BatchPredict(::testing::benchmark::State& bm_state) {
  tensorflow::testing::StopTiming();

  Network n;
  n.Load("/tmp/saved_model");

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

  tensorflow::testing::StartTiming();
  for (auto _ : bm_state) {
    auto loop = [&bm_state, &adapter, &n]() {
      NetworkFiberBatch batcher(n);
      boost::fibers::fiber prediction_loop_fiber;
      {
        std::vector<boost::fibers::fiber> fibers;
        std::vector<Mcts> mcts_vec;
        for (int i = 0; i < bm_state.range(0); ++i) {
          mcts_vec.emplace_back(batcher.new_fiber());
        }
        prediction_loop_fiber = batcher.start_prediction_fiber();
        for (auto& mcts : mcts_vec) {
          fibers.emplace_back(
              std::bind(&Mcts::Search, &mcts, std::ref(adapter), 10, ""));
        }

        std::for_each(fibers.begin(), fibers.end(),
                      std::mem_fn(&boost::fibers::fiber::join));
      }
      prediction_loop_fiber.join();
    };
    std::vector<std::thread> threads(bm_state.range(1));
    std::for_each(threads.begin(), threads.end(),
                  [&loop](std::thread& t) { t = std::thread(loop); });
    std::for_each(threads.begin(), threads.end(),
                  std::mem_fn(&std::thread::join));
  }
  tensorflow::testing::StopTiming();
  bm_state.SetItemsProcessed(n.num_pred_lines());
}

BENCHMARK(BM_BatchPredict)
    ->ArgPair(512, 1)
    ->ArgPair(512, 2)
    ->ArgPair(512, 4)
    ->ArgPair(1024, 1)
    ->ArgPair(1024, 2)
    ->ArgPair(1024, 4)
    ->ArgPair(4096, 1)
    ->ArgPair(4096, 2)
    ->ArgPair(4096, 4);

TEST(Mcts, MctsSimpleWin) {
  Network n;
  n.Load("/tmp/saved_model");

  {
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

    SnakeBoard16 state(p1, p2, [](const SnakeBoard16&) {
      return Point{15, 15};
    });
    SnakeMctsAdapter adapter(state);
    NetworkFiberBatch batcher(n);
    Direction out_direction;
    boost::fibers::fiber prediction_loop_fiber;
    {
      Mcts mcts(batcher.new_fiber());
      prediction_loop_fiber = batcher.start_prediction_fiber();
      auto d = mcts.Search(adapter, 20);
      LOG(ERROR) << Direction_Name(d);
      adapter.execute(d);
      adapter.print(false);
      d = mcts.Search(adapter, 20);
      LOG(ERROR) << Direction_Name(d);
      adapter.execute(d);
      adapter.print(false);
    }
    prediction_loop_fiber.join();
  }
}

TEST(Mcts, Starvation) {
  Network n;
  n.Load("/tmp/saved_model");

  {
    Snake p1({
        Point{5, 7},
    });

    Snake p2({
        Point{11, 9},
    });

    p1.moves_since_last_apple_ = 98;
    p2.moves_since_last_apple_ = 98;

    SnakeBoard16 state(p1, p2, [](const SnakeBoard16&) { return Point{3, 7}; });
    SnakeMctsAdapter adapter(state);
    NetworkFiberBatch batcher(n);
    Direction out_direction;
    boost::fibers::fiber prediction_loop_fiber;
    {
      Mcts mcts(batcher.new_fiber());
      prediction_loop_fiber = batcher.start_prediction_fiber();
      auto d = mcts.Search(adapter, 10000, "/home/geli/p1.dot");
      LOG(ERROR) << Direction_Name(d);
      adapter.execute(d);
      adapter.print(false);
      d = mcts.Search(adapter, 10000, "/home/geli/p2.dot");
      LOG(ERROR) << Direction_Name(d);
      adapter.execute(d);
      adapter.print(false);
    }
    prediction_loop_fiber.join();
  }
}

}  // namespace snake