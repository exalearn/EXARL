#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <stdio.h>
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

    for(unsigned int i=0; i<dimension; i++) {
        for(unsigned int j=0; j<dimension - 1; j++)
            availableMoves.push_back(line_t(i*dimension + j, i*dimension + j + 1));
    }
    for(unsigned int i=0; i<dimension - 1; i++) {
        for(unsigned int j=0; j<dimension; j++)
            availableMoves.push_back(line_t(i*dimension + j, (i+1)*dimension + j));
    } 
}

void GameBoard::printLines() {
    for (auto &it : lines)
        std::cout << it.first << ", " << it.second << std::endl;
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

bool GameBoard::checkBox(unsigned int row, unsigned int col) {
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

bool GameBoard::makeMove(line_t move) {
    auto iter = std::find(availableMoves.begin(), availableMoves.end(), move);
    if(iter != availableMoves.end()) {
        lines.push_back(move);
        availableMoves.erase(iter);
        auto temp = lookForNewBoxes();
        curScore+=temp;
        // std::cout << "Player " << (int)player << " made a move " << move.first << " " << move.second << " " << temp << std::endl;
        return (temp > 0);
    }
    return false;
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

int GameBoard::getScores() {
    return curScore;
}

int GameBoard::findNextMove(int &min, int &max, double &totalScore, int &totalGames, bool flip) {
    //Base case
    if(availableMoves.empty()) {
        int opp = ((dimension - 1) * (dimension - 1)) - curScore;
        totalGames++;
        if(player == 1) {
            min = curScore;
            max = curScore;
            totalScore+=curScore;
            // std::cout << totalScore << " ++ " << curScore << " " << opp << std::endl;
            return (curScore > opp) ? 1 : 0;
        }
        else { //player 2
            min = curScore * -1;
            max = curScore * -1;
            totalScore+=opp;
            // std::cout << totalScore << " -- " << curScore << " " << opp << std::endl;
            return (opp > curScore) ? 1 : 0;
        }
    }

    //Recursion
    int sum = 0;
    for(int i=0; i<availableMoves.size(); i++) {
        int tempMin, tempMax;
        GameBoard nextBoard(*this);
        if(flip)
            nextBoard.flipPlayer();
        sum+=nextBoard.findNextMove(tempMin, tempMax, totalScore, totalGames, !nextBoard.makeMove(availableMoves[i]));
        if(tempMax > max)
            tempMax = max;
        if(tempMin < min)
            tempMin = min;
    }
    return sum;
} 

GameBoard::line_t GameBoard::getNextMove() {
    int index = -1;
    int minIndex = -1;
    int sum = 0;
    int min = ((dimension - 1) * (dimension - 1));
    int max = 0;

    // #pragma omp parallel for
    for(unsigned int i=0; i<availableMoves.size(); i++) {
        int tempMin = 0;
        int tempMax = 0;
        double tempTotalScore = 0;
        int tempTotalGames = 0;
        
        GameBoard nextBoard(*this);
        int tempSum = nextBoard.findNextMove(tempMin, tempMax, tempTotalScore, tempTotalGames, !nextBoard.makeMove(availableMoves[i]));
        //For parallel make atomic
        if(tempMax > max)
            tempMax = max;
        if(tempMin < min) {
            tempMin = min;
            minIndex = i;
        }
        if(sum < tempSum) {
            sum = tempSum;
            index = i;
        }
    }
    if(index >= 0)
        return availableMoves[index];
    return availableMoves[minIndex];
}

double GameBoard::scoreMove(GameBoard::line_t move) {
    int min = 0;
    int max = 0;
    double totalScore = 0;
    int totalGames = 0;
    findNextMove(min, max, totalScore, totalGames, !makeMove(move));
    std::cout << totalScore << " / " << totalGames << std::endl;
    return totalScore/totalGames;
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

std::vector<int> GameBoard::serializeBoard() {
    std::vector<int> ret;
    if(lines.size() < availableMoves.size()) {
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
    }
    else {
        for(unsigned int i=0; i<dimension; i++) {
            for(unsigned int j=0; j<dimension - 1; j++) {
                auto line = line_t(i*dimension + j, i*dimension + j + 1);
                if(std::find(availableMoves.begin(), availableMoves.end(), line) == availableMoves.end())
                    ret.push_back(1);
                else
                    ret.push_back(0);
            }
        }
        for(unsigned int i=0; i<dimension - 1; i++) {
            for(unsigned int j=0; j<dimension; j++) {
                auto line = line_t(i*dimension + j, (i+1)*dimension + j);
                if(std::find(availableMoves.begin(), availableMoves.end(), line) == availableMoves.end())
                    ret.push_back(1);
                else
                    ret.push_back(0);
            }
        }
    }
    
    for(unsigned int i=0; i<dimension-1; i++) {
        for(unsigned int j=0; j<dimension-1; j++)
            ret.push_back((int)boxes[i][j]);
    }
    
    int opp = ((dimension - 1) * (dimension - 1)) - curScore;
    ret.push_back(curScore);
    ret.push_back(opp);
    return ret;
}
