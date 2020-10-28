#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <stdio.h>

class GameBoard {
    public:
        typedef std::pair<unsigned int, unsigned int> line_t;
        typedef std::vector<line_t>::iterator lineIter_t;
        
        GameBoard(unsigned int dim);
        GameBoard(const GameBoard &gameBoard);
        ~GameBoard();

        void initEmptyBoard();
        void initRandom(int moves);
        void sortLines(bool print=false);
        void printBoard();
        
        bool checkBox(int row, int col);
        unsigned int lookForNewBoxes();

        void flipPlayer();
        bool makeMove(line_t move, bool &valid);
        
        int findNextMove(int &min, int &max, double &totalScore, int &totalGames, bool flip);
        line_t getNextMove();
        void OpponentMove();

        double scoreMove(line_t move, bool &flip);
        void getScores(int &cur, int &opp);
        bool gameOver();
        
        std::vector<int> serializeBoard();

    private:
        unsigned int dimension;
        unsigned int freeBoxes;
        int curScore;
        char player;
        
        std::vector<line_t> lines;
        std::vector<line_t> availableMoves;

        char ** boxes;
};