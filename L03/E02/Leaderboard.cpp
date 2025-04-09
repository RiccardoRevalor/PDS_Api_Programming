#include "Leaderboard.h"
#include <string>

using namespace std;

//Leaderboard public functions:



void Leaderboard::addPlayer(const string &name, int score){
    //the set is ordered and automatically uses the operator <
    //I've defined the operator < for the struct Player wo it uses that operator, which compare the players' scores

    struct Player np = {name, score};
    players.insert(np);
}



void Leaderboard::removePlayer(const string &name) {

    //use an iterator to search for the right name

    for (auto it = players.begin(); it != players.end(); it++){
        if (it->PlayerName == name){
            //erase at it
            players.erase(it);
            break;
        }
    }
}


void Leaderboard::updateScore(const string &name, int newScore){

    //search for the player name and then update the player's score
    //use an iterator to search for the right name


    for (auto it = players.begin(); it != players.end(); it++){
        if (it->PlayerName == name){
            //change score
            //hoewver, objects in a set a always constant
            //so I have to remove the object and add back the new object

            players.erase(it); //remove previous version of Player
            Player newPlayer = {name, newScore}; //new version of the Player
            players.insert(newPlayer);
            break;
        }
    }
}


void Leaderboard::printTopPlayers(int n){
    //check n is positive
    if (n <= 0){
        cerr << "Error: n cannot be negative or zero!" << endl;
    }
    
    //check n is not > s.size

    if (static_cast<size_t>(n) > players.size()){
        cerr << "Error: n is too big!" << endl;
        
    }


    int counter = 0;

    for (auto it = players.begin(); it != players.end() && counter != n; it++, counter++){
        cout << "Name: " << it->PlayerName << " - Score: " << it->PlayerScore << endl;
    }
}

