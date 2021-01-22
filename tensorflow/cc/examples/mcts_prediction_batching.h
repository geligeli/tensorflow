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

}