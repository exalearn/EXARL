#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "dotsAndBoxes.h"

GameBoard::GameBoard(unsigned int dim): 
    dimension(dim),
    freeBoxes((dim-1)*(dim-1)),
    curScore(0),
    player(1) {
        boxes = new char*[dimension-1];
        for(unsigned int i=0; i<dimension-1; i++) {
            boxes[i] = new char[dimension-1];
            memset(boxes[i], 0, sizeof(char)*(dimension-1));
        }
    }

GameBoard::GameBoard(const GameBoard &gameBoard):
    dimension(gameBoard.dimension),
    freeBoxes(gameBoard.freeBoxes),
    curScore(gameBoard.curScore),
    player(gameBoard.player),
    lines(gameBoard.lines),
    availableMoves(gameBoard.availableMoves) {
        boxes = new char*[dimension-1];
        for(unsigned int i=0; i<dimension-1; i++) {
            boxes[i] = new char[dimension-1];
            memcpy(boxes[i], gameBoard.boxes[i], sizeof(char)*(dimension-1));
        }
    }

GameBoard::~GameBoard() {
    for(unsigned int i=0; i<dimension-1; i++) {
        delete boxes[i];
    }
    delete boxes;
}

void GameBoard::initEmptyBoard() {
    curScore = 0;
    freeBoxes = (dimension-1)*(dimension-1);
    player = 1;
    for(unsigned int i=0; i<dimension-1; i++) {
        memset(boxes[i], 0, sizeof(char)*(dimension-1));
    }

    availableMoves.clear();
    lines.clear();
    
    for(unsigned int i=0; i<dimension; i++) {
        for(unsigned int j=0; j<dimension - 1; j++)
            availableMoves.push_back(line_t(i*dimension + j, i*dimension + j + 1));
    }
    for(unsigned int i=0; i<dimension - 1; i++) {
        for(unsigned int j=0; j<dimension; j++)
            availableMoves.push_back(line_t(i*dimension + j, (i+1)*dimension + j));
    } 
}

void GameBoard::initRandom(int moves) {
    initEmptyBoard();

    srand (time(NULL));
    int count = 0;
    while(count < moves) {
        int dir = rand() % 2;
        int srcRow = rand() % dimension;
        int srcCol = rand() % dimension;
        
        int dstRow = -1;
        int dstCol = -1;
        
        if(dir) {
            if(srcRow + 1 < dimension) {
                dstRow = srcRow + 1;
                dstCol = srcCol;
            }
        }
        else {
            if(srcCol + 1 < dimension) {
                dstRow = srcRow;
                dstCol = srcCol + 1;
            }
        }

        if(dstRow > -1  && dstCol > -1) {
            line_t move(srcRow * dimension + srcCol, dstRow * dimension + dstCol);
            auto iter = std::find(availableMoves.begin(), availableMoves.end(), move);
            if(iter != availableMoves.end()) {
                lines.push_back(move);
                availableMoves.erase(iter);
                count++;
            }
        }
        
    }

    sortLines();
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++) {
            if(!boxes[i][j] && checkBox(i,j)) {
                boxes[i][j] = 1 + rand()%2;
                if(boxes[i][j] == 1)
                    curScore++;
            }
        }
    }
}

void GameBoard::sortLines(bool print) {
    if(print) {
        std::sort(lines.begin(), lines.end(), [&] (line_t const& a, line_t const& b) -> bool { 
                auto aSrcRow = a.first / dimension;
                auto bSrcRow = b.first / dimension;
                if(aSrcRow < bSrcRow)
                    return true;
                else if(aSrcRow == bSrcRow)
                    return a.second < b.second;
                return false;
            });
    }
    else
        std::sort(lines.begin(), lines.end());
}

void GameBoard::printBoard() {
    sortLines(true);
    auto iter = lines.begin();
    for(unsigned int i=0; i<dimension; i++) {
        //Horizontal lines
        for(unsigned int j=0; j<dimension; j++) {
            printf(".");
            if(iter != lines.end()) {
                auto line = *iter;
                auto srcRow = line.first / dimension;
                auto dstRow = line.second / dimension;
                auto srcCol = line.first % dimension;
                if(srcRow == i && dstRow == i && srcCol == j) {
                    printf("_");
                    iter++;
                }
                else
                    printf(" ");
            }
            else
                printf(" ");
        }
        printf("\n");
        //Vertical lines
        for(unsigned int j=0; j<dimension; j++) {
            if(iter != lines.end()) {
                auto line = *iter;
                auto srcRow = line.first / dimension;
                auto dstRow = line.second / dimension;
                auto srcCol = line.first % dimension;
                if(srcRow == i && dstRow == i+1 && srcCol == j) {
                    printf("|");
                    iter++;
                }
                else
                    printf(" ");
            }
            else
                printf(" ");
            
            if((i+1<dimension) && (j+1<dimension))
                printf("%u", boxes[i][j]);
        }
        printf("\n");
    }
}

bool GameBoard::checkBox(int row, int col) {
    line_t box[4] = {
        std::make_pair(row * dimension + col, row * dimension + col + 1),             //  _
        std::make_pair(row * dimension + col, (row + 1) * dimension + col),           // |
        std::make_pair(row * dimension + col + 1, (row + 1) * dimension + col + 1),   //   |
        std::make_pair((row + 1) * dimension + col, (row + 1) * dimension + col + 1)  //  _
    };
    auto it = lines.begin();
    for(unsigned int i=0; i<4; i++) {
        it = std::find(it, lines.end(), box[i]);
        // printf("Checking box: %d %d -- %d %d = %u\n", row, col, box[i].first, box[i].second, (it == lines.end()));
        if(it == lines.end())
            return false;
    }
    return true;
}

unsigned int GameBoard::lookForNewBoxes() {
    unsigned int ret = 0;
    sortLines();
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++) {
            if(!boxes[i][j] && checkBox(i,j)) {
                boxes[i][j] = player;
                ret++;
            }
        }
    }
    return ret;
}

void GameBoard::flipPlayer() {
    player++;
    if(player == 3)
        player = 1;

    curScore = 0;
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++) {
            if(boxes[i][j]==player) {
                curScore++;
            }
        }   
    }
}

bool GameBoard::makeMove(line_t move, bool &valid) {
    auto iter = std::find(availableMoves.begin(), availableMoves.end(), move);
    if(iter != availableMoves.end()) {
        valid = true;
        lines.push_back(move);
        availableMoves.erase(iter);
        auto temp = lookForNewBoxes();
        curScore+=temp;
        // std::cout << "Player " << (int)player << " made a move " << move.first << " " << move.second << " " << temp << std::endl;
        return (temp > 0);
    }
    valid = false;
    return false;
}

int GameBoard::findNextMove(int &min, int &max, double &totalScore, int &totalGames, bool flip) {
    //Base case
    if(availableMoves.empty()) {
        int opp = ((dimension - 1) * (dimension - 1)) - curScore;
        totalGames++;
        if(player == 1) {
            min = curScore;       //Always player1's score
            max = curScore;       //Always player1's score
            totalScore+=curScore; //Always player1's score
            // std::cout << totalScore << " ++ " << curScore << " " << opp << std::endl;
            if(curScore == opp)
                return 0;
            return (curScore > opp) ? 1 : -1;
        }
        else { //player 2
            min = opp;       //Always player1's score
            max = opp;       //Always player1's score
            totalScore+=opp; //Always player1's score
            // std::cout << totalScore << " -- " << curScore << " " << opp << std::endl;
            if(curScore == opp)
                return 0;
            return (opp > curScore) ? 1 : -1;
        }
    }

    //Recursion
    int sum = 0;
    bool valid;
    for(int i=0; i<availableMoves.size(); i++) {
        int tempMin, tempMax;
        GameBoard nextBoard(*this);
        if(flip)
            nextBoard.flipPlayer();
        sum+=nextBoard.findNextMove(tempMin, tempMax, totalScore, totalGames, !nextBoard.makeMove(availableMoves[i], valid));
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
    
    bool valid;

    // #pragma omp parallel for
    for(unsigned int i=0; i<availableMoves.size(); i++) {
        int tempMin = 0;
        int tempMax = 0;
        double tempTotalScore = 0;
        int tempTotalGames = 0;
        
        //Make board copy
        GameBoard nextBoard(*this);
        //Do next move and evaluate.  These will be done serially.
        int tempSum = nextBoard.findNextMove(tempMin, tempMax, tempTotalScore, tempTotalGames, !nextBoard.makeMove(availableMoves[i], valid));
        
        //For parallel make atomic
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

    if(player == 1) {
        if(maxSum > 0 && maxSumIndex >= 0) //Lets go with the most chances of winning
            return availableMoves[maxSumIndex];
        if(maxIndex >= 0) //Otherwise lets just max out the score
            return availableMoves[maxIndex];
    }
    else {
        if(minSum < 0 && minSumIndex >= 0)
            return availableMoves[minSumIndex];
        if(minIndex >= 0)
            return availableMoves[minIndex];
    }

    std::cout << "WE DON'T HAVE A VETTED MOVE..." << std::endl;
    return availableMoves[0];
}

void GameBoard::OpponentMove() {
    if(player == 1)
        flipPlayer();

    bool valid;
    if(!availableMoves.empty()) {
        auto move = getNextMove();
        while(makeMove(move, valid) && !availableMoves.empty()) {
            move = getNextMove();
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
    double totalScore = 0;
    int totalGames = 0;
    
    bool valid;
    flip = !makeMove(move, valid);
    
    if(!valid)
        return -DBL_MAX;

    findNextMove(min, max, totalScore, totalGames, flip);
    // std::cout << totalScore << " / " << totalGames << std::endl;
    return totalScore/totalGames;
}

void GameBoard::getScores(int &cur, int &opp) {
    cur = 0;
    opp = 0;
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++) {
            if(boxes[i][j]==1) {
                cur++;
            }
            else if(boxes[i][j]==2) {
                opp++;
            }
        }   
    }
}

bool GameBoard::gameOver() {
    return availableMoves.empty();
}

std::vector<int> GameBoard::serializeBoard() {
    std::vector<int> ret;
    for(unsigned int i=0; i<dimension; i++) {
        for(unsigned int j=0; j<dimension - 1; j++) {
            auto line = line_t(i*dimension + j, i*dimension + j + 1);
            if(std::find(lines.begin(), lines.end(), line) == lines.end())
                ret.push_back(0);
            else
                ret.push_back(1);
        }
    }

    for(unsigned int i=0; i<dimension - 1; i++) {
        for(unsigned int j=0; j<dimension; j++) {
            auto line = line_t(i*dimension + j, (i+1)*dimension + j);
            if(std::find(lines.begin(), lines.end(), line) == lines.end())
                ret.push_back(0);
            else
                ret.push_back(1);
        }
    }
    
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++)
            if(boxes[i][j] == 1)
                ret.push_back(1);
            else
                ret.push_back(0);    
    }
    
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++)
            if(boxes[i][j] == 2)
                ret.push_back(1);
            else
                ret.push_back(0);    
    }
    
    return ret;
}
