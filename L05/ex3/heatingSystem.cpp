#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <string>
#define TARGET_WAIT 5
#define ADMIN_WAIT 3
#define HEATING_CONTRIBUTION 0.3
#define ADMIN_THRESHOLD 0.2
#define STARTING_TEMP 18
#define STARTING_TARGET 18.5

using namespace std;

float temp; //temperature in the room
float targetTemp; //target temperature the user wants
bool running = true;
bool heatingOn = false; //at the beginning the heating system is off
mutex m;

void t_targetTemp(){
    while(running){
        {
            lock_guard<mutex> lock(m);
            string input;
            cout << "[targetTemp] Current temp: " << temp << " - Insert new targetTemp, other characters for leaving it as it is, -1 for exiting the program: " ;
            cin >> input;
            if (input.compare("-1") == 0){
                //set running to false
                running = false;
                break;
            } else if (!input.empty()){
                try {
                    float finput = stof(input);
                    //update targetTemp
                    targetTemp = finput;
                } catch(exception e){
                    cout << "[targetTemp] Invalid targetTemp" << endl;
                }
            }
        }

        //wait for 5 seconds
        this_thread::sleep_for(chrono::seconds(TARGET_WAIT));
    }

    {
        lock_guard<mutex> lock(m);
        cout << "[targetTemp] terminated." << endl;
    }
}

void t_currentTemp(){
    while(running){
        {
            lock_guard<mutex> lock(m);
            if (heatingOn){
                //add 0.3 degrees
                temp += HEATING_CONTRIBUTION;
            } else {
                temp -= HEATING_CONTRIBUTION;
            }
            cout << "[currentTemp] CurrentTemp " << temp << endl;
        }

        //wait for 5 seconds
        this_thread::sleep_for(chrono::seconds(TARGET_WAIT));
    }

    {
        lock_guard<mutex> lock(m);
        cout << "[currentTemp] terminated." << endl;
    }
}

void t_Admin(){
    while(running){
        {
            lock_guard<mutex> lock(m);
            //check temp
            if (heatingOn){
                //heating system on
                if (temp > targetTemp + ADMIN_THRESHOLD){
                    //switch off heating system
                    heatingOn = false;
                }
            } else {
                //heating system off
                if (temp <= targetTemp + ADMIN_THRESHOLD){
                    //switch on heating system
                    heatingOn = true;
                }
            }

            cout << "[Admin] CurrentTemp: " << temp << " - targetTemp: " << targetTemp << "Heating status: " << heatingOn << endl;
        }

        //wait for 3 seconds
        this_thread::sleep_for(chrono::seconds(ADMIN_WAIT));
    }

    {
        lock_guard<mutex> lock(m);
        cout << "[Admin] terminated." << endl;
    }
}


int main(void){

    //set starting temp and targetTemp
    temp = STARTING_TEMP;
    targetTemp = STARTING_TARGET;

    thread t0(t_targetTemp);
    thread t1(t_currentTemp);
    thread t2(t_Admin);

    t0.join(); t1.join(); t2.join();

    cout << "Program terminates." << endl;


    return 0;
}