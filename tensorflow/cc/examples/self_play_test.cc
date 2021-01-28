#include "tensorflow/cc/examples/self_play.h"

#include "absl/strings/str_join.h"
#include "tensorflow/cc/examples/network_fiber_batch.h"
#include "tensorflow/core/platform/test.h"

namespace snake {

TEST(SelfPlay, OneRoundTest) {
  Network n;
  n.Load("/tmp/saved_model");

  NetworkFiberBatch batcher_a(n);
  NetworkFiberBatch batcher_b(n);

  int num_games = 250;
  std::vector<SnakeMctsAdapter> states;
  boost::fibers::fiber fib1, fib2;
  {
    std::vector<boost::fibers::fiber> game_fibers;
    for (int i = 0; i < num_games; ++i) {
      game_fibers.emplace_back([a = Mcts(batcher_a.new_fiber()),
                                b = Mcts(batcher_b.new_fiber()),
                                i = i]() mutable {
        std::vector<SnakeMctsAdapter> states;
        SelfPlay(a, b, 50, &states);
        LOG(ERROR) << "game " << i << " done in n=" << states.size()
                   << " moves";
      });
    }
    fib1 = batcher_a.start_prediction_fiber();
    fib2 = batcher_b.start_prediction_fiber();
    std::for_each(game_fibers.begin(), game_fibers.end(),
                  std::mem_fn(&boost::fibers::fiber::join));
  }
  fib1.join();
  fib2.join();

  LOG(ERROR) << states.size();
}

}  // namespace snake