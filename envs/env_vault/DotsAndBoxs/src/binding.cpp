#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dotsAndBoxes.h"

GameBoard * board = NULL;
int boardSize = 3;
int numInitMoves = 5;

void reset() {
    if(!board) {
        printf("New\n");
        board = new GameBoard(boardSize);
    }
    board->initRandom(numInitMoves);
}

double step(int src, int dst) {
    if(!board)
        reset();
    
    bool endTurn = false;
    double score = board->scoreMove(GameBoard::line_t(src,dst), endTurn);
    //If you make an illegal move turn will end and you get worst score.
    if(endTurn)
        board->OpponentMove();
    return score;
}

std::vector<int> state() {
    if(!board)
        reset();
    return board->serializeBoard();
}

bool done() {
    return board->gameOver();
}

void print() {
    std::cout << "Dots and Boxes" << std::endl;
    if(board) {
        board->printBoard();
        int player1, player2;
        board->getScores(player1, player2);
        std::cout << "Player1: " << player1 << " Player2: " << player2 << std::endl;
    }
}

namespace py = pybind11;

PYBIND11_MODULE(dotsandboxes, m) {
    m.doc() = "Dots and Boxes game for Exalearn!";

    m.def("reset", &reset, "Reset");
    m.def("step", &step, "Step");
    m.def("state", &state, "State");
    m.def("print", &print, "Print");
    m.def("done", &done, "Done");
    
    m.attr("__version__") = "dev";
}