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
#include <bitset>
#include <random>
#include <thread>
#include <vector>

#include "absl/strings/str_join.h"
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
  static Point from_coord(const Coord& c) {
    return {static_cast<signed char>(c.x()), static_cast<signed char>(c.y())};
  }
};

class Snake {
 public:
  explicit Snake(Point head) : points_({head}) {}
  explicit Snake(std::vector<Point> points) : points_(points) {}

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

  int moves_since_last_apple_ = 0;

 private:
  std::vector<Point> points_;
};

template <int ARENA_SIZE>
class SnakeBoard {
 public:
  SnakeBoard()
      : SnakeBoard(Snake{Point{ARENA_SIZE / 4, ARENA_SIZE / 2}},
                   Snake{Point{3 * ARENA_SIZE / 4, ARENA_SIZE / 2}}) {}

  SnakeBoard(Snake p1, Snake p2)
      : SnakeBoard(p1, p2, std::mem_fn(&SnakeBoard::random_free_position)) {}

  SnakeBoard(Snake p1, Snake p2,
             std::function<Point(const SnakeBoard&)> apple_spawner)
      : p1_(p1), p2_(p2), pixels_{}, apple_spawner_(apple_spawner) {
    for (auto p : p1_.points()) {
      pixels_[p.x + ARENA_SIZE * p.y] = Pixel::P1;
    }
    for (auto p : p2_.points()) {
      pixels_[p.x + ARENA_SIZE * p.y] = Pixel::P2;
    }
    spawn_apple();
  }

  struct PlayerView {
    const Snake& player;
    const Snake& opponent;
    const SnakeBoard& board;
    bool valid_move(Direction d) const {
      return has_move() ? board.is_unoccupied(player.peek(d)) : true;
    }
    bool has_move() const {
      for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
        if (board.is_unoccupied(player.peek(static_cast<Direction>(i)))) {
          return true;
        }
      }
      return false;
    }
  };

  SnakeBoard(PlayerView pv)
      : game_state_(pv.board.game_state_),
        p1_(pv.player),
        p2_(pv.opponent),
        apple_position_(pv.board.apple_position_),
        pixels_{},
        apple_spawner_(std::mem_fn(&SnakeBoard::random_free_position)) {
    for (auto p : p1_.points()) {
      pixels_[p.x + ARENA_SIZE * p.y] = Pixel::P1;
    }
    for (auto p : p2_.points()) {
      pixels_[p.x + ARENA_SIZE * p.y] = Pixel::P2;
    }
    at(apple_position_) = Pixel::APPLE;
  }

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
        game_state_ = GameState::DRAW;
        return game_state_;
      } else if (p1_.size() > p2_.size()) {
        game_state_ = GameState::P1_WIN;
        return game_state_;
      }
      game_state_ = GameState::P2_WIN;
      return game_state_;
    }

    if (p1_alive != p2_alive) {
      game_state_ = p1_alive ? GameState::P1_WIN : GameState::P2_WIN;
      return game_state_;
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

    game_state_ = GameState::RUNNING;
    return game_state_;
  }

  void print(bool clear_screen = true) const {
    // 🟣🟤🟢🟡🟠⚪⚫🔴🔵
    if (clear_screen) {
      std::cout << "\x1B[2J\x1B[H";
    } else {
      std::cout << "\n----------------------\n";
    }
    // std::cout << "\n";
    for (int y = 0; y < ARENA_SIZE; ++y) {
      for (int x = 0; x < ARENA_SIZE; ++x) {
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

    std::cout << "P1 🟦 time left to live: "
              << 100 - p1_.moves_since_last_apple() << "\n";
    std::cout << "P2 🟩 time left to live: "
              << 100 - p2_.moves_since_last_apple() << "\n";
    switch (game_state_) {
      case GameState::P1_WIN:
        std::cout << "P1 🟦 is the winner" << std::endl;
        break;
      case GameState::P2_WIN:
        std::cout << "P2 🟩 is the winner" << std::endl;
        break;
      case GameState::DRAW:
        std::cout << "It's a draw" << std::endl;
        break;
      default:
        break;
    }
  }

  std::string html_table() const {
    std::vector<std::string> lines;
    // lines.push_back("<table>");
    for (int y = 0; y < ARENA_SIZE; ++y) {
      lines.push_back("<tr>");
      for (int x = 0; x < ARENA_SIZE; ++x) {
        lines.push_back("<td fixedsize=\"true\" width=\"24\" height=\"24\"");

        switch (pixels_[x + ARENA_SIZE * y]) {
          case EMPTY:
            lines.push_back(" ");
            break;
          case P1:
            lines.push_back(" bgcolor=\"blue\"");
            break;
          case P2:
            lines.push_back(" bgcolor=\"green\"");
            break;
          case APPLE:
            lines.push_back(" bgcolor=\"red\"");
            break;
        }
        if ((p1_.head().x == x && p1_.head().y == y) ||
            (p2_.head().x == x && p2_.head().y == y)) {
          lines.push_back(">H</td>");
        } else {
          lines.push_back("></td>");
        }
      }
      lines.push_back("</tr>");
    }

    lines.push_back(absl::StrCat("<tr><td colspan=\"", ARENA_SIZE, "\">"));
    switch (game_state_) {
      case GameState::P1_WIN:
        lines.push_back("blue wins");
        break;
      case GameState::P2_WIN:
        lines.push_back("green wins");
        break;
      case GameState::DRAW:
        lines.push_back("draw");
        break;
      default:
        lines.push_back("running");
        break;
    }
    lines.push_back(absl::StrCat(" P1(", 100 - p1_.moves_since_last_apple(),
                                 ") P2(", 100 - p2_.moves_since_last_apple(),
                                 ")"));
    lines.push_back("</td></tr>");

    return absl::StrJoin(lines, "");
  }

  Point random_free_position() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(0,
                                                  ARENA_SIZE * ARENA_SIZE - 1);
    size_t pos = distrib(gen);
    const size_t initial_pos = pos;
    while (Pixel::EMPTY != pixels_[pos]) {
      ++pos;
      if (pos >= pixels_.size()) {
        pos = 0;
      }
      CHECK(pos != initial_pos);
    }
    return {static_cast<signed char>(pos % ARENA_SIZE),
            static_cast<signed char>(pos / ARENA_SIZE)};
  }

  void spawn_apple() {
    if (apple_spawner_) {
      apple_position_ = apple_spawner_(*this);
      at(apple_position_) = Pixel::APPLE;
    }
  }

  Point apple_position() const { return apple_position_; }

  bool is_terminal() const { return game_state_ != GameState::RUNNING; }

  GameState game_state() const { return game_state_; }

  constexpr int size() const { return ARENA_SIZE; }

 private:
  GameState game_state_ = GameState::RUNNING;
  Snake p1_;
  Snake p2_;
  Point apple_position_;
  std::array<Pixel, ARENA_SIZE * ARENA_SIZE> pixels_;
  std::function<Point(const SnakeBoard&)> apple_spawner_;
};

using SnakeBoard16 = SnakeBoard<16>;

using StratFun = std::function<Direction(const SnakeBoard16::PlayerView&)>;

GameState RunGame(StratFun a, StratFun b) {
  SnakeBoard16 board;
  GameState retval = board.move(a(board.p1_view()), b(board.p2_view()));
  while (retval == GameState::RUNNING) {
    // std::this_thread::sleep_for(std::chrono::milliseconds(200));
    retval = board.move(a(board.p1_view()), b(board.p2_view()));
    board.print();
  }
  return retval;
}

Direction GreedyStrategy(const SnakeBoard16::PlayerView& p) {
  struct DirectionScore {
    Direction dir;
    bool unoccupied;
    int apple_dist;
  };
  std::array<DirectionScore, Direction_ARRAYSIZE> scores;
  for (int i = Direction_MIN; i < Direction_ARRAYSIZE; ++i) {
    const Direction d = static_cast<Direction>(i);
    auto new_head = p.player.head().peek(d);
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

  bool valid_action(Direction) const {
    return true;
    // if (move_queued_) {
    //   return state_.p2_view().valid_move(d);
    // }
    // return state_.p1_view().valid_move(d);
  }

  bool is_terminal() const {
    return state_.is_terminal();  // || (valid_actions().count() == 0);
  }

  int player() const { return move_queued_ ? 1 : -1; }

  void print(bool clear_screen = true) const {
    state_.print(clear_screen);
    if (move_queued_) {
      std::cout << "p1 queue move: " << Direction_Name(p1_queued_move_)
                << std::endl;
    } else {
      std::cout << "no move queued for p1" << std::endl;
    }
  }
  std::string html(std::string info) const {
    std::vector<std::string> lines;
    lines.push_back("<table border=\"1\">");
    lines.push_back(state_.html_table());
    lines.push_back("<tr><td colspan=\"16\">");
    if (move_queued_) {
      lines.push_back(
          absl::StrCat("p1 queue move: ", Direction_Name(p1_queued_move_)));
    } else {
      lines.push_back("p1 no move queued");
    }

    lines.push_back(
        absl::StrCat("</td></tr><tr><td colspan=\"", state_.size(), "\">"));
    lines.push_back(info);
    lines.push_back("</td></tr></table>");
    return absl::StrJoin(lines, "");
  }

  const SnakeBoard16& board() const { return state_; }

 private:
  SnakeBoard16 state_;
  Direction p1_queued_move_;
  bool move_queued_ = false;
};

}  // namespace snake

#endif
