#include "tensorflow/cc/examples/snake.h"

#include <gtest/gtest.h>

namespace snake {

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

TEST(Snake, SnakeTest) {
  Snake snake(Point{3, 4});
  snake.grow(Direction::LEFT);
  ASSERT_EQ(2, snake.points().size());
  EXPECT_EQ((Point{2, 4}), snake.points()[0]);
  EXPECT_EQ((Point{3, 4}), snake.points()[1]);
}

TEST(SnakeBoard, SnakeBoardTest) {
  SnakeBoard16 board;
  board.move(Direction::LEFT, Direction::RIGHT);
  board.print();
}

TEST(SnakeBoard, RunGame) { RunGame(GreedyStrategy, GreedyStrategy); }

}  // namespace snake