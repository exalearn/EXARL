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
        void printLines();
        void sortLines(bool print=false);
        bool checkBox(unsigned int row, unsigned int col);
        unsigned int lookForNewBoxes();
        bool makeMove(line_t move);
        void flipPlayer();
        int getScores();
        int findNextMove(int &min, int &max, double &totalScore, int &totalGames, bool flip);
        line_t getNextMove();
        double scoreMove(line_t move);
        std::vector<int> serializeBoard();
        void printBoard();

    private:
        unsigned int dimension;
        unsigned int freeBoxes;
        int curScore;
        char player;
        
        std::vector<line_t> lines;
        std::vector<line_t> availableMoves;

        char ** boxes;
};