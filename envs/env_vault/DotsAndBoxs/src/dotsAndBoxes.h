#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <stdio.h>

class GameBoard {
    public:
        typedef int line_t;
        typedef std::vector<line_t>::iterator lineIter_t;
        
        GameBoard(unsigned int dim);
        GameBoard(const GameBoard &gameBoard);
        ~GameBoard();

        void initEmptyBoard();
        void initRandom(int moves);
        void printBoard();
        
        bool setLine(line_t move, bool horizontal);
        bool checkLine(line_t move, bool horizontal);
        bool checkBox(int box);
        unsigned int lookForNewBoxes(line_t move);

        void flipPlayer();
        bool makeMove(line_t move, bool &valid);
        
        int findNextAvailableMove(int &start);
        int findNextAvailableMoveFromIndex(unsigned int index);
        int findNextMove(int &min, int &max, int &totalScore, int &totalGames, bool flip);
        line_t getNextMoveMPI();
        line_t getNextMove();
        void OpponentMove(bool MPI=false);

        double scoreMove(line_t move, bool &flip);
        void getScores(int &cur, int &opp);
        bool gameOver();
        
        std::vector<int> serializeBoard();
        void deserializeBoard(std::vector<int> state);
        unsigned int serialBoardSize();

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
};