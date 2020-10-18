#include "dotsAndBoxes.h"

int main(int argc, char* argv[]) {
    std::cout << "hello world" << std::endl;
    GameBoard empty(3);
    empty.initEmptyBoard();
    empty.makeMove(GameBoard::line_t(0,1));
    empty.makeMove(GameBoard::line_t(1,2));
    empty.makeMove(GameBoard::line_t(0,3));
    empty.makeMove(GameBoard::line_t(2,5));
    empty.makeMove(GameBoard::line_t(3,6));
    empty.makeMove(GameBoard::line_t(5,8));
    empty.makeMove(GameBoard::line_t(6,7));
    empty.makeMove(GameBoard::line_t(7,8));
    empty.makeMove(GameBoard::line_t(1,4));
    empty.sortLines();
    // empty.printLines();
    empty.printBoard();
    auto score = empty.scoreMove(GameBoard::line_t(3,4));
    empty.flipPlayer();
    empty.makeMove(GameBoard::line_t(4,7));
    std::cout << score << std::endl;
    // auto next = empty.getNextMove();
    // empty.makeMove(next);
    empty.printBoard();
    auto vec = empty.serializeBoard();
    for(auto it: vec) {
        printf("%d ", it);
    }
    printf("\n");
    return 0;
}