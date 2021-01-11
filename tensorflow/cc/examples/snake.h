#ifndef TENSORFLOW_CC_EXAMPLES_SNAKE_H_
#define TENSORFLOW_CC_EXAMPLES_SNAKE_H_

/*
            UP
            -y
            |
            |
LEFT -x ---------  +x  RIGHT
            |
            |
            +y
           DOWN
*/

#include <array>
#include <random>
#include <thread>
#include <vector>

#include "tensorflow/cc/examples/snake.pb.h"
#include "tensorflow/core/platform/default/logging.h"
// #include "tensorflow/cc/examples/snake.grpc.pb.h"

namespace snake {

struct Point {
  signed char x;
  signed char y;
  int mdist(Point other) const {
    return std::abs(x - other.x) + std::abs(y - other.y);
  }
  Point peek(Direction d) const {
    Point result = *this;
    switch (d) {
      case UP:
        --result.y;
        break;
      case DOWN:
        ++result.y;
        break;
      case LEFT:
        --result.x;
        break;
      case RIGHT:
        ++result.x;
        break;
    }
    return result;
  }
  bool operator==(Point other) const { return x == other.x && y == other.y; }
  Coord to_coord() const {
    Coord r;
    r.set_x(x);
    r.set_y(y);
    return r;
  }
  static Point from_coord(const Coord& c) { return {c.x(), c.y()}; }
};

class Snake {
 public:
  explicit Snake(Point head) : points_({head}) {}

  Point move(Direction d) {
    Point result = points_.back();
    points_.insert(points_.begin(), points_.front().peek(d));
    points_.pop_back();
    ++moves_since_last_apple_;
    return result;
  }

  void grow(Direction d) {
    points_.insert(points_.begin(), points_.front().peek(d));
    moves_since_last_apple_ = 0;
  }

  Point head() const { return *points_.cbegin(); }

  Point peek(Direction d) const { return points_.cbegin()->peek(d); }

  const std::vector<Point>& points() const { return points_; };

  size_t size() const { return points_.size(); }

  int moves_since_last_apple() const { return moves_since_last_apple_; }

 private:
  std::vector<Point> points_;
  int moves_since_last_apple_ = 0;
};

template <int ARENA_SIZE>
class SnakeBoard {
 public:
  SnakeBoard()
      : SnakeBoard(Snake{Point{ARENA_SIZE / 4, ARENA_SIZE / 2}},
                   Snake{Point{3 * ARENA_SIZE / 4, ARENA_SIZE / 2}}) {}

  SnakeBoard(Snake p1, Snake p2)
      : p1_(p1),
        p2_(p2),
        pixels_{},
        apple_spawner(std::bind(&SnakeBoard::random_free_position, this)) {
    for (auto p : p1_.points()) {
      pixels_[p.x + ARENA_SIZE * p.y] = Pixel::P1;
    }
    for (auto p : p2_.points()) {
      pixels_[p.x + ARENA_SIZE * p.y] = Pixel::P2;
    }
    spawn_apple();
  }

  struct PlayerView {
    const Snake& player1;
    const Snake& player2;
    const SnakeBoard& board;
  };

  PlayerView p1_view() const { return {p1_, p2_, *this}; }

  PlayerView p2_view() const { return {p2_, p1_, *this}; }

  Pixel& at(Point p) { return pixels_[p.x + ARENA_SIZE * p.y]; }

  Pixel at(Point p) const { return pixels_[p.x + ARENA_SIZE * p.y]; }

  bool is_oob(Point p) const {
    return p.x < 0 || p.y < 0 || p.x >= ARENA_SIZE || p.y >= ARENA_SIZE;
  }

  bool is_empty(Point p) const { return !is_oob(p) && at(p) == Pixel::EMPTY; }

  bool is_unoccupied(Point p) const {
    return !is_oob(p) && (at(p) == Pixel::EMPTY || at(p) == Pixel::APPLE);
  }

  GameState move(Direction p1, Direction p2) {
    auto p1next = p1_.peek(p1);
    auto p2next = p2_.peek(p2);

    const bool p1_alive =
        is_unoccupied(p1next) && p1_.moves_since_last_apple() < 100;
    const bool p2_alive =
        is_unoccupied(p2next) && p2_.moves_since_last_apple() < 100;

    if (p1next == p2next || (!p1_alive && !p2_alive)) {
      if (p1_.size() == p2_.size()) {
        return GameState::DRAW;
      } else if (p1_.size() > p2_.size()) {
        return GameState::P1_WIN;
      }
      return GameState::P2_WIN;
    }

    if (p1_alive != p2_alive) {
      return p1_alive ? GameState::P1_WIN : GameState::P2_WIN;
    }

    bool apple_consumed = false;
    // No headon collision case.
    if (at(p1next) == Pixel::APPLE) {
      apple_consumed = true;
      p1_.grow(p1);
    } else {
      at(p1_.move(p1)) = Pixel::EMPTY;
    }

    if (at(p2next) == Pixel::APPLE) {
      apple_consumed = true;
      p2_.grow(p2);
    } else {
      at(p2_.move(p2)) = Pixel::EMPTY;
    }

    at(p1next) = Pixel::P1;
    at(p2next) = Pixel::P2;

    if (apple_consumed) {
      spawn_apple();
    }

    return GameState::RUNNING;
  }

  void print() const {
    // 🟣🟤🟢🟡🟠⚪⚫🔴🔵
    std::cout << "\x1B[2J\x1B[H";
    for (int x = 0; x < ARENA_SIZE; ++x) {
      for (int y = 0; y < ARENA_SIZE; ++y) {
        switch (pixels_[x + ARENA_SIZE * y]) {
          case EMPTY:
            std::cout << "⬜";
            break;
          case P1:
            std::cout << "🟦";
            break;
          case P2:
            std::cout << "🟩";
            break;
          case APPLE:
            std::cout << "🍎";
            break;
        }
      }
      std::cout << std::endl;
    }
    // using namespace std::chrono;
    // milliseconds ms =
    //     duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    // std::cout << ms.count() << std::endl;
  }

  Point random_free_position() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, ARENA_SIZE * ARENA_SIZE - 1);
    int pos = distrib(gen);
    const int initial_pos = pos;
    while (Pixel::EMPTY != pixels_[pos]) {
      ++pos;
      if (pos >= pixels_.size()) {
        pos = 0;
      }
      CHECK(pos != initial_pos);
    }
    return {pos % ARENA_SIZE, pos / ARENA_SIZE};
  }

  void spawn_apple() {
    if (apple_spawner) {
      apple_position_ = apple_spawner();
      at(apple_position_) = Pixel::APPLE;
    }
  }

  Point apple_position() const { return apple_position_; }

 private:
  Snake p1_;
  Snake p2_;
  Point apple_position_;
  std::array<Pixel, ARENA_SIZE * ARENA_SIZE> pixels_;
  std::function<Point()> apple_spawner;
};

using SnakeBoard16 = SnakeBoard<16>;

using StratFun = std::function<Direction(const SnakeBoard16::PlayerView&)>;

GameState RunGame(StratFun a, StratFun b) {
  SnakeBoard16 board;
  GameState retval = board.move(a(board.p1_view()), b(board.p2_view()));
  while (retval == GameState::RUNNING) {
    board.print();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    retval = board.move(a(board.p1_view()), b(board.p2_view()));
  }
  return retval;
}

Direction GreedyStrategy(const SnakeBoard16::PlayerView& p) {
  struct DirectionScore {
    Direction dir;
    int apple_dist;
    bool unoccupied;
  };
  std::array<DirectionScore, Direction_ARRAYSIZE> scores;
  for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
    const Direction d = static_cast<Direction>(i);
    auto new_head = p.player1.head().peek(d);
    scores[i].unoccupied = p.board.is_unoccupied(new_head);
    scores[i].dir = d;
    scores[i].apple_dist = new_head.mdist(p.board.apple_position());
  }
  return std::max_element(scores.begin(), scores.end(),
                          [](DirectionScore a, DirectionScore b) {
                            // return a<b
                            if (a.unoccupied != b.unoccupied) {
                              return b.unoccupied;
                            }
                            return a.apple_dist > b.apple_dist;
                          })
      ->dir;
}

}  // namespace snake

#endif
