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

DotsAndBoxes::DotsAndBoxes(unsigned int dim):
    horizontalLines(NULL),
    verticalLines(NULL),
    player1Boxes(NULL),
    player2Boxes(NULL),
    dimension(dim),
    availableMoves(2 * dim * (dim-1)),
    player1Score(0),
    player2Score(0),
    nextIndex(0),
    player(1),
    perspective(1) {
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

DotsAndBoxes::DotsAndBoxes(const DotsAndBoxes &DotsAndBoxes):
    horizontalLines(NULL),
    verticalLines(NULL),
    player1Boxes(NULL),
    player2Boxes(NULL),
    dimension(DotsAndBoxes.dimension),
    availableMoves(DotsAndBoxes.availableMoves),
    player1Score(DotsAndBoxes.player1Score),
    player2Score(DotsAndBoxes.player2Score),
    nextIndex(DotsAndBoxes.nextIndex),
    player(DotsAndBoxes.player), 
    perspective(DotsAndBoxes.perspective) {
        auto numBoxes = (dimension-1) * (dimension-1);
        auto boxArraySize = numBoxes/64 + numBoxes%64;
        player1Boxes = new uint64_t[boxArraySize];
        player2Boxes = new uint64_t[boxArraySize];

        memcpy(player1Boxes, DotsAndBoxes.player1Boxes, sizeof(uint64_t)*boxArraySize);
        memcpy(player2Boxes, DotsAndBoxes.player2Boxes, sizeof(uint64_t)*boxArraySize);

        auto numLinesPerDir = dimension * (dimension-1);
        auto linesArraySize = numLinesPerDir/64 + numLinesPerDir%64;
        horizontalLines = new uint64_t[linesArraySize];
        verticalLines = new uint64_t[linesArraySize];

        memcpy(horizontalLines, DotsAndBoxes.horizontalLines, sizeof(uint64_t)*linesArraySize);
        memcpy(verticalLines, DotsAndBoxes.verticalLines, sizeof(uint64_t)*linesArraySize);
    }

DotsAndBoxes::~DotsAndBoxes() {
    delete player1Boxes;
    delete player2Boxes;
    delete horizontalLines;
    delete verticalLines;
}

void DotsAndBoxes::initEmptyBoard() {
    availableMoves = 
    player1Score = 0;
    player2Score = 0;
    player = 1;
    perspective = 1;
    availableMoves = 2 * dimension * (dimension-1);
    nextIndex = 0;

    auto numBoxes = (dimension-1) * (dimension-1);
    auto boxArraySize = numBoxes/64 + numBoxes%64;
    memset(player1Boxes, 0, sizeof(uint64_t)*boxArraySize);
    memset(player2Boxes, 0, sizeof(uint64_t)*boxArraySize);

    auto numLinesPerDir = dimension * (dimension-1);
    auto linesArraySize = numLinesPerDir/64 + numLinesPerDir%64;
    memset(horizontalLines, 0, sizeof(uint64_t)*linesArraySize);
    memset(verticalLines, 0, sizeof(uint64_t)*linesArraySize);
}

bool DotsAndBoxes::setLine(int move, bool horizontal) {
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

bool DotsAndBoxes::checkLine(int move, bool horizontal) {
    int index = move / 64;
    int offset = move % 64;
    uint64_t mask = 1UL << offset;
    
    if(horizontal)
        return (horizontalLines[index] & mask);
    else
        return (verticalLines[index] & mask);
}

bool DotsAndBoxes::checkBox(int box) {
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

unsigned int DotsAndBoxes::lookForNewBoxes(int move) {
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

bool DotsAndBoxes::makeMove(int move, bool &valid) {
    bool ret = false;
    valid = false;
    auto numLinesPerDir = dimension * (dimension-1);
    if(move < numLinesPerDir) { //Horizontal
        valid = setLine(move, true);
    }
    else if(move < 2 * numLinesPerDir) { //Vertical
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

bool DotsAndBoxes::player1() {
    return (player == 1);
}

void DotsAndBoxes::flipPlayer() {
    player = (player == 1) ? 2 : 1;
}

void DotsAndBoxes::flipPerspective() {
    perspective = (perspective == 1) ? 2 : 1;
}

void DotsAndBoxes::initRandom(int moves, bool MPI) {
    initEmptyBoard();
    unsigned int seed = time(NULL);
    if(MPI)
        MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    srand(seed);
    int count = 0;
    int failed = 0;
    bool valid;
    while(count < moves && availableMoves) {
        int move = rand() % (2 * dimension * (dimension-1));
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

void DotsAndBoxes::printBoard() {
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

int DotsAndBoxes::numAvailableMoves() { return availableMoves; }

int DotsAndBoxes::findNextAvailableMove(int &start) {
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

int DotsAndBoxes::findNextAvailableMoveFromIndex(unsigned int index) {
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

bool DotsAndBoxes::terminal() {
    if(availableMoves) {
        int winningBoxes = (dimension-1) * (dimension-1) / 2;
        if(player1Score > winningBoxes || player2Score > winningBoxes)
            return true;
        return false;
    }
    return true;
}

#define getMax(a,b) ((a) > (b) ? (a) : (b))
#define getMin(a,b) ((a) < (b) ? (a) : (b))

int DotsAndBoxes::findNextMove(int &alpha, int &beta, bool flip) {
    //Base case
    if(terminal()) {
        if(perspective == 1) {
            return player1Score;
        }
        else {
            return player2Score;
        }
    }

    //Recursion
    bool valid;
    int moveStart = 0;
    bool maxNode = (perspective == player);
    int value = (maxNode) ? INT_MIN : INT_MAX;
    if(maxNode) {
        for(int i=0; i<availableMoves; i++) {
            DotsAndBoxes nextBoard(*this);
            if(flip)
                nextBoard.flipPlayer();
            auto move = findNextAvailableMove(moveStart);
            value = getMax(value, nextBoard.findNextMove(alpha, beta, !nextBoard.makeMove(move, valid)));
            alpha = getMax(value, alpha);
            if(alpha >= beta)
                break;
        }
    }
    else {
        for(int i=0; i<availableMoves; i++) {
            DotsAndBoxes nextBoard(*this);
            if(flip)
                nextBoard.flipPlayer();
            auto move = findNextAvailableMove(moveStart);
            value = getMin(value, nextBoard.findNextMove(alpha, beta, !nextBoard.makeMove(move, valid)));
            beta = getMin(value, beta);
            if(beta <= alpha)
                break;
        }
    }
    return value;
} 

int DotsAndBoxes::getNextMoveOMP(int start, int end, int &value) {
    value = (perspective==player) ? INT_MIN : INT_MAX;
    if(!terminal()) {
        int index = -1;
        #pragma omp parallel for shared(index, value)
        for(unsigned int i=start; i<end; i++) {
            //Thread private variables
            bool valid;
            int alpha = INT_MIN;
            int beta = INT_MAX;
            DotsAndBoxes nextBoard(*this);

            //Do next move and evaluate.  These will be done serially.
            auto move = findNextAvailableMoveFromIndex(i);
            int tempValue = nextBoard.findNextMove(alpha, beta, !nextBoard.makeMove(move, valid));
            
            #pragma omp critical
            {
                if(perspective==player) {
                    if(tempValue > value) {
                        value = tempValue;
                        index = i;
                    }
                }
                else {
                    if(tempValue < value) {
                        value = tempValue;
                        index = i;
                    }
                }
            }
        }
        return findNextAvailableMoveFromIndex(index);
    }
    std::cout << "Board is terminal..." << std::endl;
    return -1;
}

int DotsAndBoxes::getNextMoveMPI(int numNodes, int &value) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ret = -1;
    value = (perspective==player) ? INT_MIN : INT_MAX;
    if(!terminal()) {
        int results[2];
        int boardsPerRank = availableMoves / numNodes;
        int rem = availableMoves % numNodes;

        std::vector<int> serial = serializeBoard();
        MPI_Bcast(&serial[0], serial.size(), MPI_INT, 0, MPI_COMM_WORLD);
        
        if(rank)
            deserializeBoard(serial);

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
        results[0] = getNextMoveOMP(startEnd[0], startEnd[1], results[1]);

        int * recvBuf = NULL;
        if(!rank)
            recvBuf = new int[2*numNodes];
        MPI_Gather(results, 2, MPI_INT, recvBuf, 2, MPI_INT, 0, MPI_COMM_WORLD);
        
        if(!rank) {
            for(int i=0; i<numNodes; i++) {
                int * tempRes = &recvBuf[i*2];
                if(tempRes[1] > results[1])
                    results[0] = tempRes[0];
            }
            ret = results[0];
            delete sendBuf;
            delete recvBuf;
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    fflush(stdout);
    return ret;
}

int DotsAndBoxes::getNextMove(int &score, bool MPI) {
    int numNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    if(MPI && availableMoves > omp_get_max_threads())
        return getNextMoveMPI(numNodes, score);

    return getNextMoveOMP(0, availableMoves, score);
}

void DotsAndBoxes::OpponentMove() {
    if(player == 1)
        flipPlayer();
    if(perspective == 1)
        flipPerspective();

    if(!terminal()) {
        bool valid;
        int value;
        auto move = getNextMove(value);
        while(makeMove(move, valid) && !terminal()) {
            move = getNextMove(value);
        }
        //Give it back to player1
        flipPlayer();
        flipPerspective();
    }
}

int DotsAndBoxes::scoreMove(int move, bool &flip) {
    //Only score player1 moves
    if(player == 2)
        flipPlayer();
    if(perspective == 2)
        flipPerspective();

    bool valid;
    flip = !makeMove(move, valid);

    if(!valid) //Is is a move that is already made
        return -INT_MAX;

    if(terminal()) //Is the game over?
        return player1Score;

    if(flip) //Do we get another turn or no?
        flipPlayer();

    int value; //Lets get the max score after our move
    getNextMove(value);
    return value;
}

void DotsAndBoxes::getScores(int &cur, int &opp) {
    cur=player1Score;
    opp=player2Score;
}

std::vector<int> DotsAndBoxes::serializeBoard() {
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

    if(player == 1)
        ret.push_back(1);
    else
        ret.push_back(0);

    if(perspective == 1)
        ret.push_back(1);
    else
        ret.push_back(0);

    return ret;
}

void DotsAndBoxes::deserializeBoard(std::vector<int> state) {
    initEmptyBoard();
    auto iter = state.begin();
    auto numLinesPerDir = dimension * (dimension-1);
    for(int i=0; i<numLinesPerDir; i++) {
        if(*iter) {
            setLine(i, true);
            availableMoves--;
        }
        iter++;
    }

    for(int i=0; i<numLinesPerDir; i++) {
        if(*iter) {
            setLine(i, false);
            availableMoves--;
        }
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

    if(*iter)
        player = 1;
    else
        player = 2;
    
    iter++;
    if(*iter)
        perspective = 1;
    else
        perspective = 2;
    iter++;
}
