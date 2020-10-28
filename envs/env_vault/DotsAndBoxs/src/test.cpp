#include "dotsAndBoxes.h"
#include <omp.h>
#define NANOSECS 1000000000

uint64_t globalTimeStamp(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    uint64_t timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}

GameBoard * board = NULL;
int boardSize = 3;
int numInitMoves = 5;

int main(int argc, char* argv[]) {
    if(argc == 3) {
        boardSize = atoi(argv[1]);
        numInitMoves = atoi(argv[2]);
    }
    
    auto initStart = globalTimeStamp();
    board = new GameBoard(boardSize);
    board->initRandom(numInitMoves);
    auto initEnd = globalTimeStamp();
    printf("Game Board: %d Init Moves: %d Time: %lf sec\n", boardSize, numInitMoves, (double)(initEnd-initStart)/1e9);

    int action = rand();
    int numLines = 2 * boardSize * (boardSize - 1);
    int spaceLen = numLines + 2 * (boardSize - 1) * (boardSize - 1);
    int src, dst;
    if(action < numLines/2) {
        auto row = action / (boardSize - 1);
        auto col = action % (boardSize - 1);
        src = row * boardSize + col;
        dst = src + 1;
    }
    else {
        action -= numLines/2;
        auto row = action / boardSize;
        auto col = action % boardSize;
        src = row * boardSize + col;
        dst = src + boardSize;
    }


    auto scoreStart = globalTimeStamp();
    bool endTurn = false;
    double score = board->scoreMove(GameBoard::line_t(src,dst), endTurn);
    auto scoreEnd = globalTimeStamp();
    printf("Player 1 Move Time: %lf\n", (double)(scoreEnd-scoreStart)/1e9);
    
    auto stateUpdateStart = globalTimeStamp();
    if(endTurn)
        board->OpponentMove();
    auto stateUpdateEnd = globalTimeStamp();
    printf("Player 2 Move Time: %lf\n", (double)(stateUpdateEnd-stateUpdateStart)/1e9);

    auto stateStart = globalTimeStamp();
    board->serializeBoard();
    board->gameOver();
    auto stateEnd = globalTimeStamp();
    printf("State Time: %lf\n", (double)(stateEnd-stateStart)/1e9);

    auto shutdownStart = globalTimeStamp();
    delete board;
    auto shutdownEnd = globalTimeStamp();
    printf("Game Board: %d Init Moves: %d Time: %lf sec\n", boardSize, numInitMoves, (double)(shutdownEnd-shutdownStart)/1e9);

    return 0;
}