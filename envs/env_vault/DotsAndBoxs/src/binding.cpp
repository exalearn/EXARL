#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dotsAndBoxes.h"

DotsAndBoxes * board = NULL;
int boardSize = 3;
int numInitMoves = 5;

void setParams(int boardDimension, int numberOfInitialMoves) {
    boardSize = boardDimension;
    numInitMoves = numberOfInitialMoves;
}

void reset() {
    if(!board)
        board = new DotsAndBoxes(boardSize);
    board->initRandom(numInitMoves);
}

int step(int move) {
    if(!board)
        reset();

    int score = INT_MIN;
    bool valid;
    bool flip = !board->makeMove(move, valid);
    if(valid) {
        if(flip) {
            board->flipPlayer();
            board->flipPerspective();
        }

        int player1, player2;
        board->getScores(player1, player2);
        if(board->player1())
            score = player1;
        else
            score = player2;
    }
    return score;
}

std::vector<int> state() {
    if(!board)
        reset();
    return board->serializeBoard();
}

bool done() {
    if(board->terminal()) {
        delete board;
        board = NULL;
        return true;
    }
    return false;
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

    m.def("setParams", &setParams, "Sets the board size and number of initial moves.");
    m.def("reset", &reset, "Reset");
    m.def("step", &step, "Step");
    m.def("state", &state, "State");
    m.def("print", &print, "Print");
    m.def("done", &done, "Done");
    
    m.attr("__version__") = "dev";
}