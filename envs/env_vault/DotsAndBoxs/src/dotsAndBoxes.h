#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <stdio.h>

class GameBoard {
    public:
        GameBoard(unsigned int dim);
        GameBoard(const GameBoard &gameBoard);
        ~GameBoard();

        void initEmptyBoard();
        void initRandom(int moves);
        void printBoard();
        
        bool setLine(int move, bool horizontal);
        bool checkLine(int move, bool horizontal);
        bool checkBox(int box);
        unsigned int lookForNewBoxes(int move);
        bool terminal();

        void flipPlayer();
        void flipPerspective();
        bool makeMove(int move, bool &valid);
        
        int findNextAvailableMove(int &start);
        int findNextAvailableMoveFromIndex(unsigned int index);
        int findNextMove(int &alpha, int &beta, bool flip);
        int getNextMove(int &score);
        int getNextMoveMPI(int numNodes, int &value);
        int getNextMoveOMP(int start, int end,int &value);
        void OpponentMove();

        int scoreMove(int move, bool &flip);
        void getScores(int &cur, int &opp);
        
        std::vector<int> serializeBoard();
        void deserializeBoard(std::vector<int> state);

    private:
        uint64_t * horizontalLines;
        uint64_t * verticalLines;
        uint64_t * player1Boxes;
        uint64_t * player2Boxes;
        unsigned int dimension;
        int availableMoves;
        int player1Score;
        int player2Score;
        char player;
        char perspective;
};