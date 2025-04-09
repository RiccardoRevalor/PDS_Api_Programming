#ifndef LEADERBOARD_H
#define LEADERBOARD_H

#include <string>
#include <iostream>
#include <set>

using namespace std;


struct Player {
    string PlayerName;
    int PlayerScore;


    //Constructor
    Player(const string &name, int score) : PlayerName(name), PlayerScore(score) {};

    //Operator < to be used for marking comparisons within the players in the set
    //It's important to define the operator as a CONST operator because the set wants the exact same operator definition
    bool operator< (const Player &other) const{
        //descending order based on the scores

        //if scores are the same, use lexicographical order
        if (PlayerScore == other.PlayerScore){
            return PlayerName > other.PlayerName;
        }

        return PlayerScore > other.PlayerScore;
    }


};


class Leaderboard{

    public:
        void addPlayer(const string &name, int score);
        void removePlayer(const string &name);
        void updateScore(const string &name, int newScore);
        void printTopPlayers(int n);


    private:
        //set to store the players in an ordered way
        set<struct Player> players;


};


#endif