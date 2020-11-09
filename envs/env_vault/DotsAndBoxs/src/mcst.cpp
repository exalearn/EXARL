#include <iostream>
#include <stdio.h>
#include <stdlib.h>

template<class T>
class MCTSNode {
    private:
        T * state;
        unsigned int wins;
        unsigned int loses;
        unsigned int numVisits;
        unsigned int toExplore;
        MCTSNode * parent;
        MCTSNode * children;

        void setState(T * board);
        int q();
        int n();

    public:
        MCTSNode(T board, MCTSNode * node);
        ~MCTSNode();

        bool terminal();
        bool isExpanded();
        MCTSNode * expand();
        unsigned int rolloutPolicy();
        int rollout();
        void backpropagate(int result);
        unsigned int bestChild();
};

template<class T>
MCTSNode<T>::MCTSNode(T * board, MCTSNode * node): state(board), wins(0), loses(0), numVisits(0), toExplore(0), parent(node), children(NULL) { 
    if(!board->terminal()) {
        toExplore = board->numAvailableMoves();
        childern = new MCTSNode<T>[board->numAvailableMoves()];
    }
}

template<class T>
MCTSNode<T>::MCTSNode(): state(NULL), wins(0), loses(0), numVisits(0), toExplore(0), parent(node), children(NULL) { 
    if(!board->terminal()) {
        toExplore = board->numAvailableMoves();
        childern = new MCTSNode<T>[board->numAvailableMoves()];
    }
}

template<class T>
MCTSNode<T>::~MCTSNode() {
    if(children)
        delete children;
}

template<class T>
void MCTSNode<T>::setStart(T * board) { state = board; }

template<class T>
int MCTSNode<T>::q() { return wins - loses; }

template<class T>
int MCTSNode<T>::n() { return numVisits; }

template<class T>
bool MCTSNode<T>::terminal() { return state->terminal(); }

template<class T>
bool MCTSNode<T>::isExpanded() { return toExplore == state->numAvaialbleMoves(); }

template<class T>
MCTSNode<T> * MCTSNode<T>::expand() {
    bool valid;
    int move = board->findNextAvailableMove(nextMove);
    T newBoard(*board);
    if(!newBoard.makeMove(move, &valid)
        newBoard.flip();
    child[toExplore].setBoard(newBoard);
    auto ret = child[toExplore];
    toExplore++;
    return ret;
}

template<class T>
unsigned int MCTSNode<T>::rolloutPolicy() { 
    unsigned int move = rand();
    // return possible_moves[np.random.randint(len(possible_moves))]
}

template<class T>
int MCTSNode<T>::rollout() { 
    // current_rollout_state = self.state
    // while not current_rollout_state.is_game_over():
    //     possible_moves = current_rollout_state.get_legal_actions()
    //     action = self.rollout_policy(possible_moves)
    //     current_rollout_state = current_rollout_state.move(action)
    // return current_rollout_state.game_result

}

template<class T>
void MCTSNode<T>::backpropagate(int result) { 
    numVisits += 1;
    // self._results[result] += 1.
    if(parent)
        parent->backpropagate(result);
}

template<class T>
unsigned int MCTSNode<T>::bestChild() { }

// template<class T>
// class MCTS {
//     public:
//         MCTS();
//         ~MCTS();

//         unsigned int search(unsigned int numSim);
//         unsigned int treePolicy(Node node);
        
//     private:
//         T root;
// };

// template<class T>
// unsigned int search(unsigned int numSim) {
//     for(unsigned int i=0; i<simulations_number; i++) {            
//         v = treePolicy();
//         reward = v.rollout()
//         v.backpropagate(reward)
//     }
//     return bestChild();
// }

// template<class T>
// unsigned int treePolicy() {
//     auto current = root;
//     while(!current.terminal()) {
//         if(!current.fullyExpanded())
//             current.expand();
//         else
//             current = current.bestChild();
//     }
//     return current;
// }

int main(void) {
    return 0;
}