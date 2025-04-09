#include "Leaderboard.h"
#include <string>
#include <iostream>

using namespace std;


int main(void){

    //Create a new Leaderboard
    Leaderboard L;




    L.addPlayer("Zapata", 10);
    L.addPlayer("Ricci", 9);
    L.addPlayer("Adams", 20);
    L.addPlayer("Coco", 7);

    L.printTopPlayers(2);

    cout << endl << endl;

    L.updateScore("Ricci", 21);

    L.printTopPlayers(2);

    cout << endl << endl;


    L.removePlayer("Ricci");

    L.printTopPlayers(3);

    cout << endl << endl;



    return 0;
}