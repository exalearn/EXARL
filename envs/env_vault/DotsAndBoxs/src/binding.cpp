#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dotsAndBoxes.h"

GameBoard * board = NULL;

void dotsAndBoxesReset() {
    if(board)
        board->initEmptyBoard();
    else
        board = new GameBoard(3);
    
    board->makeMove(GameBoard::line_t(0,1));
    board->makeMove(GameBoard::line_t(1,2));
    board->makeMove(GameBoard::line_t(0,3));
    board->makeMove(GameBoard::line_t(2,5));
    board->makeMove(GameBoard::line_t(3,6));
    board->makeMove(GameBoard::line_t(5,8));
    board->makeMove(GameBoard::line_t(6,7));
    board->makeMove(GameBoard::line_t(7,8));
    board->makeMove(GameBoard::line_t(1,4));
    board->sortLines();
}

double dotsAndBoxesStep(unsigned int src, unsigned int dst) {
    if(!board)
        dotsAndBoxesReset();
    return board->scoreMove(GameBoard::line_t(src,dst));
}

std::vector<int> dotsAndBoxesState() {
    if(!board)
        dotsAndBoxesReset();
    return board->serializeBoard();
}

namespace py = pybind11;

PYBIND11_MODULE(DotsAndBoxes, m) {
    m.doc() = "Dot and Boxes game api"; // optional module docstring
    m.def("dotsAndBoxesReset", &dotsAndBoxesReset, "Resets a game");
}