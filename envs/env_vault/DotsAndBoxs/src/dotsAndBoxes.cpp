#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include <mpi.h>
#include "dotsAndBoxes.h"

GameBoard::GameBoard(unsigned int dim): 
    horizontalLines(NULL),
    verticalLines(NULL),
    player1Boxes(NULL),
    player2Boxes(NULL),
    dimension(dim),
    availableMoves(2 * dim * (dim-1)),
    player1Score(0),
    player2Score(0),
    player(1) {
        auto numBoxes = (dim-1) * (dim-1);
        auto boxArraySize = numBoxes/64 + numBoxes%64;
        player1Boxes = new uint64_t[boxArraySize];
        player2Boxes = new uint64_t[boxArraySize];

        memset(player1Boxes, 0, sizeof(uint64_t)*boxArraySize);
        memset(player2Boxes, 0, sizeof(uint64_t)*boxArraySize);

        auto numLinesPerDir = dim * (dim-1);
        auto linesArraySize = numLinesPerDir/64 + numLinesPerDir%64;
        horizontalLines = new uint64_t[linesArraySize];
        verticalLines = new uint64_t[linesArraySize];

        memset(horizontalLines, 0, sizeof(uint64_t)*linesArraySize);
        memset(verticalLines, 0, sizeof(uint64_t)*linesArraySize);
    }

GameBoard::GameBoard(const GameBoard &gameBoard):
    horizontalLines(NULL),
    verticalLines(NULL),
    player1Boxes(NULL),
    player2Boxes(NULL),
    dimension(gameBoard.dimension),
    availableMoves(gameBoard.availableMoves),
    player1Score(gameBoard.player1Score),
    player2Score(gameBoard.player2Score),
    player(gameBoard.player) {
        auto numBoxes = (dimension-1) * (dimension-1);
        auto boxArraySize = numBoxes/64 + numBoxes%64;
        player1Boxes = new uint64_t[boxArraySize];
        player2Boxes = new uint64_t[boxArraySize];

        memcpy(player1Boxes, gameBoard.player1Boxes, sizeof(uint64_t)*boxArraySize);
        memcpy(player2Boxes, gameBoard.player2Boxes, sizeof(uint64_t)*boxArraySize);

        auto numLinesPerDir = dimension * (dimension-1);
        auto linesArraySize = numLinesPerDir/64 + numLinesPerDir%64;
        horizontalLines = new uint64_t[linesArraySize];
        verticalLines = new uint64_t[linesArraySize];

        memcpy(horizontalLines, gameBoard.horizontalLines, sizeof(uint64_t)*linesArraySize);
        memcpy(verticalLines, gameBoard.verticalLines, sizeof(uint64_t)*linesArraySize);
    }

GameBoard::~GameBoard() {
    delete player1Boxes;
    delete player2Boxes;
    delete horizontalLines;
    delete verticalLines;
}

void GameBoard::initEmptyBoard() {
    availableMoves = 
    player1Score = 0;
    player2Score = 0;
    player = 1;
    availableMoves = 2 * dimension * (dimension-1);

    auto numBoxes = (dimension-1) * (dimension-1);
    auto boxArraySize = numBoxes/64 + numBoxes%64;
    memset(player1Boxes, 0, sizeof(uint64_t)*boxArraySize);
    memset(player2Boxes, 0, sizeof(uint64_t)*boxArraySize);

    auto numLinesPerDir = dimension * (dimension-1);
    auto linesArraySize = numLinesPerDir/64 + numLinesPerDir%64;
    memset(horizontalLines, 0, sizeof(uint64_t)*linesArraySize);
    memset(verticalLines, 0, sizeof(uint64_t)*linesArraySize);
}

bool GameBoard::setLine(line_t move, bool horizontal) {
    int index = move / 64;
    int offset = move % 64;
    uint64_t mask = 1UL << offset;
    if(horizontal) {
        if(!(horizontalLines[index] & mask)) {
            horizontalLines[index] |= mask;
            return true;
        }
    }
    else {
        if(!(verticalLines[index] & mask)) {
            verticalLines[index] |= mask;
            return true;
        }
    }
    return false;
}

bool GameBoard::checkLine(line_t move, bool horizontal) {
    int index = move / 64;
    int offset = move % 64;
    uint64_t mask = 1UL << offset;
    
    if(horizontal)
        return (horizontalLines[index] & mask);
    else
        return (verticalLines[index] & mask);
}

bool setBox(int box) {
    
    return false;
}

bool GameBoard::checkBox(int box) {
    // printf("Checking %d\n", box);
    int horz = box;
    if(!checkLine(horz, true))
        return false;
    if(!checkLine(horz + dimension-1, true))
        return false;

    int row = box / (dimension-1);
    int col = box % (dimension-1);
    int vert = row * dimension + col;
    if(!checkLine(vert, false))
        return false;
    if(!checkLine(vert + 1, false))
        return false;

    int index = box / 64;
    int offset = box % 64;
    uint64_t mask = 1UL << offset;
    if(!(player1Boxes[index] & mask) && !(player2Boxes[index] & mask))
    if(player == 1) {
        // printf("Player 1 score!\n");
        player1Boxes[index] |= mask;
    }
    else {
        // printf("Player 2 score!\n");
        player2Boxes[index] |= mask;
    }

    return true;
}

unsigned int GameBoard::lookForNewBoxes(line_t move) {
    unsigned int score = 0;
    auto numLinesPerDir = dimension * (dimension-1);
    if(move < numLinesPerDir) {
        auto horMove = move;
        int row = horMove / (dimension-1);
        int col = horMove % (dimension-1);
        if(row - 1 >= 0 && checkBox((row-1) * (dimension-1) + col)) //Up
            score++;
        if(row + 1 < dimension && checkBox(row * (dimension-1) + col)) //Down
            score++;
    }
    else {
        auto vertMove = move - numLinesPerDir;
        int row = vertMove / dimension;
        int col = vertMove % dimension;
        if(col - 1 >= 0 && checkBox(row * (dimension-1) + col - 1)) // Left
            score++;
        if(col + 1 < dimension && checkBox(row * (dimension-1) + col)) //Right
            score++;
    }
    return score;
}

bool GameBoard::makeMove(line_t move, bool &valid) {
    bool ret = false;
    auto numLinesPerDir = dimension * (dimension-1);
    if(move < numLinesPerDir) { //Horizontal
        valid = setLine(move, true);
    }
    else { //Vertical
        auto temp = move - numLinesPerDir;
        valid = setLine(temp, false);
    }

    if(valid) {
        availableMoves--;
        auto temp = lookForNewBoxes(move);
        if(player == 1)
            player1Score+=temp;
        else
            player2Score+=temp;
        return (temp > 0);
    }

    return false;
}

void GameBoard::flipPlayer() {
    player = (player == 1) ? 2 : 1;
}

void GameBoard::initRandom(int moves) {
    initEmptyBoard();

    srand (time(NULL));
    int count = 0;
    int failed = 0;
    bool valid;
    while(count < moves && availableMoves) {
        line_t move = rand() % (2 * dimension * (dimension-1));
        makeMove(move, valid);
        if(valid) {
            count++;
            flipPlayer();
        }
        else
            failed++;
    }
    // printf("Good: %u Bad: %u\n", count, failed);
}

void GameBoard::printBoard() {
    for(int i=0; i<dimension-1; i++) {
        for(int j=0; j<dimension-1; j++) {
            printf(".");
            if(checkLine(i * (dimension-1) + j, true))
                printf("_", i * (dimension-1) + j);
            else
                printf(" ");
        }
        printf(".\n");
        for(int j=0; j<dimension; j++) {
            if(checkLine(i * (dimension) + j, false))
                printf("|", j);
            else
                printf(" ");

            if(j+1<dimension) {
                int box = i*(dimension-1) + j;
                int index = box / 64;
                int offset = box % 64;
                uint64_t mask = 1UL << offset;
                if(player1Boxes[index] & mask)
                    printf("1");
                else if(player2Boxes[index] & mask)
                        printf("2");
                else
                    printf(" ");
            }
        }
        printf("\n");
    }
    for(int j=0; j<dimension-1; j++) {
        printf(".");
        if(checkLine((dimension-1) * (dimension-1) + j, true))
            printf("_");
        else
            printf(" ");
    }
    printf(".\n");
}

int GameBoard::findNextAvailableMove(int &start) {
    auto numLinesPerDir = dimension * (dimension-1);
    int vertStart = 0;
    if(start<numLinesPerDir) {
        for(int i=start; i<numLinesPerDir; i++) {
            if(!checkLine(i, true)) {
                start = i + 1;
                return i;
            }
        }
    }
    else
        vertStart = start - numLinesPerDir;

    for(int i=vertStart; i<numLinesPerDir; i++) {
        if(!checkLine(i, false)) {
            start = numLinesPerDir + i + 1;
            return start - 1;
        }
    }
    printf("No moves...\n");
    return -1;
}

int GameBoard::findNextAvailableMoveFromIndex(unsigned int index) {
    unsigned int count = 0;
    auto numLinesPerDir = dimension * (dimension-1);
    for(int i=0; i<numLinesPerDir; i++) {
        if(!checkLine(i, true)) {
            if(count == index)
                return i;
            count++;
        }
    }

    for(int i=0; i<numLinesPerDir; i++) {
        if(!checkLine(i, false)) {
            if(count == index)
                return numLinesPerDir + i;
            count++;
        }
    }
    return -1;
}

int GameBoard::findNextMove(int &min, int &max, int &totalScore, int &totalGames, bool flip) {
    //Base case
    if(!availableMoves) {
        totalGames++;
        if(player == 1) {
            min = player1Score;       //Always player1's score
            max = player1Score;       //Always player1's score
            totalScore+=player1Score; //Always player1's score
        }
        else { //player 2
            min = player2Score;       //Always player2's score
            max = player2Score;       //Always player2's score
            totalScore+=player2Score; //Always player2's score
        }
        if(player1Score == player2Score)
            return 0;
        return (player1Score > player2Score) ? 1 : -1;
    }

    //Recursion
    int sum = 0;
    bool valid;
    int moveStart = 0;
    for(int i=0; i<availableMoves; i++) {
        int tempMin, tempMax;
        GameBoard nextBoard(*this);
        if(flip)
            nextBoard.flipPlayer();
        auto move = findNextAvailableMove(moveStart);
        sum+=nextBoard.findNextMove(tempMin, tempMax, totalScore, totalGames, !nextBoard.makeMove(move, valid));
        if(tempMax > max)
            max = tempMax;
        if(tempMin < min)
            min = tempMin;
    }
    return sum;
} 

GameBoard::line_t GameBoard::getNextMove() {
    int maxSumIndex = -1;
    int minSumIndex = -1;
    
    int minIndex = -1;
    int maxIndex = -1;

    int maxSum = INT_MIN;
    int minSum = INT_MAX;

    int min = INT_MAX;
    int max = INT_MIN;

    #pragma omp parallel for shared(maxSumIndex, minSumIndex, minIndex, maxIndex, maxSum, minSum, min, max)
    for(unsigned int i=0; i<availableMoves; i++) {
        int tempMin = 0;
        int tempMax = 0;
        int tempTotalScore = 0;
        int tempTotalGames = 0;
        bool valid;
        //Make board copy
        GameBoard nextBoard(*this);
        //Do next move and evaluate.  These will be done serially.
        auto move = findNextAvailableMoveFromIndex(i);
        int tempSum = nextBoard.findNextMove(tempMin, tempMax, tempTotalScore, tempTotalGames, !nextBoard.makeMove(move, valid));
        
        #pragma omp critical
        {
            if(tempMax > max) {
                max = tempMax;
                maxIndex = i;
            }

            if(tempMin < min) {
                min = tempMin;
                minIndex = i;
            }

            if(maxSum < tempSum) {
                maxSum = tempSum;
                maxSumIndex = i;
            }

            if(minSum < tempSum) {
                minSum = tempSum;
                minSumIndex = i;
            }
        }
    }

    if(player == 1) {
        if(maxSum > 0 && maxSumIndex >= 0) //Lets go with the most chances of winning
            return findNextAvailableMoveFromIndex(maxSumIndex);
        if(maxIndex >= 0) //Otherwise lets just max out the score
            return findNextAvailableMoveFromIndex(maxIndex);
    }
    else {
        if(minSum < 0 && minSumIndex >= 0)
            return findNextAvailableMoveFromIndex(minSumIndex);
        if(minIndex >= 0)
            return findNextAvailableMoveFromIndex(minIndex);
    }

    std::cout << "WE DON'T HAVE A VETTED MOVE..." << std::endl;
    return findNextAvailableMoveFromIndex(0);
}

void GameBoard::OpponentMove(bool MPI) {
    if(player == 1)
        flipPlayer();
    bool valid;
    if(availableMoves) {
        if(MPI) {
            auto move = getNextMoveMPI();
            while(makeMove(move, valid) && availableMoves) {
                move = getNextMoveMPI();
            }
        }
        else {
            auto move = getNextMove();
            printf("Player 2 move: %d\n", move);
            while(makeMove(move, valid) && availableMoves) {
                // printBoard();
                move = getNextMove();
                printf("Player 2 move: %d\n", move);
            }
            // printBoard();
        }
    }
    //Give it back to player1
    flipPlayer();
}

double GameBoard::scoreMove(GameBoard::line_t move, bool &flip) {
    //Only score player1 moves
    if(player == 2)
        flipPlayer();

    int min = 0;
    int max = 0;
    int totalScore = 0;
    int totalGames = 0;
    
    bool valid;
    flip = !makeMove(move, valid);
    
    if(!valid)
        return -DBL_MAX;

    findNextMove(min, max, totalScore, totalGames, flip);
    // std::cout << totalScore << " / " << totalGames << std::endl;
    return (double)totalScore/totalGames;
}

void GameBoard::getScores(int &cur, int &opp) {
    cur=player1Score;
    opp=player2Score;
}

bool GameBoard::gameOver() {
    return !availableMoves;
}

std::vector<int> GameBoard::serializeBoard() {
    std::vector<int> ret;
    
    auto numLinesPerDir = dimension * (dimension-1);
    for(int i=0; i<numLinesPerDir; i++) {
        if(checkLine(i, true))
            ret.push_back(1);
        else
            ret.push_back(0);
    }

    for(int i=0; i<numLinesPerDir; i++) {
        if(checkLine(i, false))
            ret.push_back(1);
        else
            ret.push_back(0);
    }
    
    auto numBoxes = (dimension-1) * (dimension-1);
    for(int i=0; i<numBoxes; i++) {
        int index = i / 64;
        int offset = i % 64;
        uint64_t mask = 1UL << offset;
        if(player1Boxes[index] & mask)
            ret.push_back(1);
        else
            ret.push_back(0);
    }

    for(unsigned int i=0; i<numBoxes; i++) {
         int index = i / 64;
        int offset = i % 64;
        uint64_t mask = 1UL << offset;
        if(player2Boxes[index] & mask)
            ret.push_back(1);
        else
            ret.push_back(0);   
    }
    return ret;
}

void GameBoard::deserializeBoard(std::vector<int> state) {
    initEmptyBoard();
    auto iter = state.begin();
    auto numLinesPerDir = dimension * (dimension-1);
    for(int i=0; i<numLinesPerDir; i++) {
        if(*iter)
            setLine(i, true);
        iter++;
    }

    for(int i=0; i<numLinesPerDir; i++) {
        if(*iter)
            setLine(i, false);
        iter++;
    }
    
    auto numBoxes = (dimension-1) * (dimension-1);
    for(int i=0; i<numBoxes; i++) {
        if(*iter) {
            int index = i / 64;
            int offset = i % 64;
            uint64_t mask = 1UL << offset;
            player1Boxes[index] |= mask;
            player1Score++;
        }
        iter++;
    }

    for(unsigned int i=0; i<numBoxes; i++) {
        if(*iter) {
            int index = i / 64;
            int offset = i % 64;
            uint64_t mask = 1UL << offset;
            player2Boxes[index] |= mask;
            player2Score++;
        }
        iter++;
    }
}

unsigned int GameBoard::serialBoardSize() {
    unsigned int numLinesPerDir = dimension * (dimension-1);
    unsigned int numBoxes = (dimension-1) * (dimension-1);
    return 2 * numLinesPerDir + 2 * numBoxes;
}

GameBoard::line_t GameBoard::getNextMoveMPI() {
    int ret = -1;
    int results[8] = {-1, -1, -1, -1, INT_MIN, INT_MAX, INT_MAX, INT_MIN};
    // {maxSumIndex, minSumIndex, minIndex, maxIndex, maxSum, minSum, min, max}

    int numNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int tempAvail = availableMoves;
    int boardsPerRank = availableMoves / numNodes;
    int rem = availableMoves % numNodes;

    std::vector<int> serial = serializeBoard();
    // if(!rank) {
    //     printf("RANK: %d\n", rank);
    //     printBoard();
    // }
    MPI_Bcast(&serial[0], serial.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if(rank) {
        deserializeBoard(serial);
        // printf("RANK: %d\n", rank);
        // printBoard();
    }

    int startEnd[2];
    int * sendBuf = NULL;
    if(!rank) {
        sendBuf = new int[numNodes*2];
        int next = 0;
        for(int i=0; i<numNodes; i++) {
            sendBuf[i*2] = next;

            next+=boardsPerRank;
            if(rem) {
                next++;
                rem--;
            }

            sendBuf[i*2 + 1] = next;
        }
    }
    MPI_Scatter(sendBuf, 2, MPI_INT, startEnd, 2, MPI_INT, 0, MPI_COMM_WORLD);
    printf("RANK: %d %d %d\n", rank, startEnd[0], startEnd[1]);
    #pragma omp parallel for shared(results)
    for(unsigned int i=startEnd[0]; i<startEnd[1]; i++) {
        int tempMin = 0;
        int tempMax = 0;
        int tempTotalScore = 0;
        int tempTotalGames = 0;
        bool valid;
        //Make board copy
        GameBoard nextBoard(*this);
        //Do next move and evaluate.  These will be done serially.
        auto move = findNextAvailableMoveFromIndex(i);
        int tempSum = nextBoard.findNextMove(tempMin, tempMax, tempTotalScore, tempTotalGames, !nextBoard.makeMove(move, valid));
        
        #pragma omp critical
        {
            // {maxSumIndex, minSumIndex, minIndex, maxIndex, maxSum, minSum, min, max}
            // {          0,           1,        2,        3,      4,      5,   6,   7}

            // if(tempMax > max) {
            if(tempMax > results[7]) {
                // max = tempMax;
                results[7] = tempMax;
                // maxIndex = i;
                results[3] = i;
            }

            // if(tempMin < min) {
            if(tempMin < results[6]) {
                // min = tempMin;
                results[6] = tempMin;
                // minIndex = i;
                results[2] = i;
            }

            // if(maxSum < tempSum) {
            if(results[4] < tempSum) {
                // maxSum = tempSum;
                results[4] = tempSum;
                // maxSumIndex = i;
                results[0] = i;
            }

            // if(minSum < tempSum) {
            if(results[5] < tempSum) {
                // minSum = tempSum;
                results[5] = tempSum;
                // minSumIndex = i;
                results[1] = i;
            }
            printf("Rank %d -- %d %d %d %d %d %d %d %d\n", rank, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]);
        }
    }

    int * recvBuf = NULL;
    if(!rank)
        recvBuf = new int[8*numNodes];
    MPI_Gather(results, 8, MPI_INT, recvBuf, 8, MPI_INT, 0, MPI_COMM_WORLD);
    if(!rank) {
        for(int i=0; i<numNodes; i++) {
            int * tempRes = &recvBuf[i*8];
            printf("%d %d %d %d %d %d %d %d\n", tempRes[0], tempRes[1], tempRes[2], tempRes[3], tempRes[4], tempRes[5], tempRes[6], tempRes[7]);
            if(tempRes[7] > results[7]) {
                results[7] = tempRes[7];
                results[3] = tempRes[3];
            }

            if(tempRes[6] < results[6]) {
                results[6] = tempRes[6];
                results[2] = tempRes[2];
            }

            if(results[4] < tempRes[4]) {
                results[4] = tempRes[4];
                results[0] = tempRes[0];
            }

            if(results[5] < tempRes[5]) {
                results[5] = tempRes[5];
                results[1] = tempRes[1];
            }
        }
    
        // {maxSumIndex, minSumIndex, minIndex, maxIndex, maxSum, minSum, min, max}
        // {          0,           1,        2,        3,      4,      5,   6,   7}
        if(player == 1) {
            if(results[4] > 0 && results[0] >= 0) //Lets go with the most chances of winning
                ret = findNextAvailableMoveFromIndex(results[0]);
            if(results[3] >= 0) //Otherwise lets just max out the score
                ret = findNextAvailableMoveFromIndex(results[3]);
        }
        else {
            if(results[5] < 0 && results[1] >= 0)
                ret = findNextAvailableMoveFromIndex(results[1]);
            if(results[2] >= 0)
                ret = findNextAvailableMoveFromIndex(results[2]);
        }

        delete sendBuf;
        delete recvBuf;
    }
    printf("Before--- %d\n", rank);
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank: %d After--- %d\n", rank, ret);
    if(ret < 0) {
        std::cout << rank << " WE DON'T HAVE A VETTED MOVE..." << std::endl;
        return findNextAvailableMoveFromIndex(0);
    }
    return ret;
}