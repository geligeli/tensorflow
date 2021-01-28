#include <boost/coroutine2/coroutine.hpp>
#include <boost/fiber/all.hpp>
#include <chrono>
#include <list>

namespace snake {
template <typename INPUT, typename OUTPUT>
struct Batching {
  struct Work {
    INPUT input;
    boost::fibers::promise<OUTPUT> r;
  };

  using ChannelType = boost::fibers::unbuffered_channel<Work>;

  struct Closer {
    void operator()(ChannelType* c) {
      //      LOG(ERROR) << "closing channel";
      c->close();
    }
  };
  using Base = std::unique_ptr<ChannelType, Closer>;
  struct BatchFiber : public Base {
    using Base::Base;
    OUTPUT BatchProcess(INPUT input) const {
      Work w;
      w.input = input;
      auto f = w.r.get_future();
      // LOG(ERROR) << "pushing to channel";
      Base::get()->push(std::move(w));
      // LOG(ERROR) << "getting response";
      return f.get();
    }
  };
  BatchFiber new_fiber() {
    channels_.emplace_back();
    return BatchFiber{&channels_.back(), Closer()};
  }
  typename boost::coroutines2::coroutine<Work>::pull_type get_batch() {
    return typename boost::coroutines2::coroutine<Work>::pull_type(
        [this](typename boost::coroutines2::coroutine<Work>::push_type& sink) {
          //          LOG(ERROR) << channels_.size();
          for (auto it = channels_.begin(); it != channels_.end();) {
            Work w;
            //            LOG(ERROR) << "Attempting to pop";
            if (it->pop(w) != boost::fibers::channel_op_status::success) {
              //              LOG(ERROR) << "releasing channel";
              it = channels_.erase(it);
              continue;
            } else {
              ++it;
            }
            // LOG(ERROR) << "channel to sink";
            sink(std::move(w));
          }
          // LOG(ERROR) << channels_.size();
        });
  }

 private:
  std::list<ChannelType> channels_;
};

}  // namespace snake