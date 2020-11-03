#include "dotsAndBoxes.h"
#include "mpi.h"
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
    MPI_Init(NULL, NULL);
    if(argc == 3) {
        boardSize = atoi(argv[1]);
        numInitMoves = atoi(argv[2]);
    }
    
    auto initStart = globalTimeStamp();
    board = new GameBoard(boardSize);
    board->initRandom(numInitMoves);
    auto initEnd = globalTimeStamp();
    // printf("Game Board: %d Init Moves: %d Time: %lf sec\n", boardSize, numInitMoves, (double)(initEnd-initStart)/1e9);

    // GameBoard check(boardSize);
    // check.deserializeBoard(board->serializeBoard());
    // board->printBoard();
    
    // printf("Start-------------------------\n");
    // board->printBoard();

    bool endTurn = false;
    auto scoreStart = globalTimeStamp();
    // int action = rand() % 2 * boardSize * (boardSize-1);
    for(int i=0; i<2 * boardSize * (boardSize-1); i++) {
        if(board->scoreMove(GameBoard::line_t(i), endTurn) > -1)
            break;
    }
    auto scoreEnd = globalTimeStamp();
    // printf("Player 1 Move Time: %lf\n", (double)(scoreEnd-scoreStart)/1e9);
    
    // printf("Player 1-------------------------\n");
    // board->printBoard();

    auto stateUpdateStart = globalTimeStamp();
    // if(endTurn)
        board->OpponentMove(true);
    auto stateUpdateEnd = globalTimeStamp();
    // printf("Player 2 Move Time: %lf\n", (double)(stateUpdateEnd-stateUpdateStart)/1e9);

    // printf("Player 2-------------------------\n");
    // board->printBoard();

    auto stateStart = globalTimeStamp();
    board->serializeBoard();
    board->gameOver();
    auto stateEnd = globalTimeStamp();
    // printf("State Time: %lf\n", (double)(stateEnd-stateStart)/1e9);

    printf("Done-------------------------\n");
    board->printBoard();

    auto shutdownStart = globalTimeStamp();
    delete board;
    auto shutdownEnd = globalTimeStamp();
    // printf("Game Board: %d Init Moves: %d Time: %lf sec\n", boardSize, numInitMoves, (double)(shutdownEnd-shutdownStart)/1e9);

    printf("Init: %lf Move: %lf Opp: %lf State: %lf Shutdown: %lf\n", (double)(initEnd-initStart)/1e9, (double)(scoreEnd-scoreStart)/1e9, (double)(stateUpdateEnd-stateUpdateStart)/1e9, (double)(stateEnd-stateStart)/1e9, (double)(shutdownEnd-shutdownStart)/1e9);
    MPI_Finalize();
    return 0;
}