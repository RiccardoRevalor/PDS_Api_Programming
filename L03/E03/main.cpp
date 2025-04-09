#include <iostream>
#include <fstream>
#include <set>
#include <map>



using namespace std;

struct User {
    string name;
    string surname;

    bool operator< (const User &other) const{
        //< const operator used by both map and set to order elements @ insertion
        //first compare name w/ other.name
        //then compare surname w/ other.surname
        //std::tie
        return tie(name, surname) > tie(other.name, other.surname);
    }


};

ostream &operator<< (ostream &output, const User &user){
    //operator << to print Users on output stream using cout
    //it's like the toString() method in java custom classes
    //usually it's common practice to define these operators outside of structs
    output << user.name << " " << user.surname;
    return output;
}

int main(void){

    //adjacency list
    map<User, set<User>> adjList; //keys are the vertices of the graph.. on the set all the adjancencies are stored


    ifstream input("socialNetwork.txt");

    if (!input){
        cerr << "Error opening the input file." << endl;
        return 1;
    }

    
    string sX, nX, sY, nY;
    string X, Y;
    string sep = "_";

    while (input >> X >> Y){
        size_t pos0 = X.find(sep); //index when the sep is
        sX = X.substr(0, pos0); //take the surname of X
        nX = X.substr(pos0 + 1); //take the name of X

        size_t pos1 = Y.find(sep);
        sY = Y.substr(0, pos1);
        nY = Y.substr(pos1 + 1);

        /*
        cout << sX << " " << nX << endl;
        cout << sY << " " << nY << endl;
        */

        User userX = {nX, sX};
        User userY = {nY, sY};

        //check if the map already contains userX
        if (adjList.find(userX) != adjList.end()){
            //map already has userX
            //add userY to their adjancencies
            adjList[userX].insert(userY);
        } else {
            //map doesn't have userX yet
            //create a new user set
            adjList[userX] = {userY}; 
        }

        //check if the map already contains userY
        if (adjList.find(userY) != adjList.end()){
            //map already has userY
            //add userX to their adjancencies
            adjList[userY].insert(userX);
        } else {
            //map doesn't have userY yet
            //create a new user set
            adjList[userY] = {userX}; 
        }


    }


    //display the list of users in lexicopraphical order
    //iterare over key, which are already ordered
    cout << "Current Users:" << endl;
    for (auto it = adjList.begin(); it != adjList.end(); it++){
        cout << it->first << endl;
    }

    //given an individual's name, display of their friends in lexicographical order
    string inputName;

    cout << "Insert a user name: ";
    cin >> inputName;

    for (auto it = adjList.begin(); it != adjList.end(); it++){
        if (it->first.name == inputName){
            //display all friends
            for (auto fr = it->second.begin(); fr != it->second.end(); fr++){
                cout << fr->name << " " << fr->surname << endl;
            }
        }
    }


    return 0;
}


