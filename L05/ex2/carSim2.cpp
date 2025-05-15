#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#define NUM_SPOTS 3
#define NUM_CARS 10

using namespace std;


mutex m;
int simDuration = -1; //starting from firs car that enters
int occupied_spots = 0;
condition_variable cv;
bool stop = false; //stop sim
chrono::steady_clock::time_point timeStart;

void threadArrival(){
    random_device rd; 
    mt19937 gen(rd()); //set seed generator
    uniform_int_distribution<> distr(1, 3); //define uniform distributuon for the range [1; 3]

    for (int i = 0; i < NUM_CARS; i++){
        //represent each carr arriving every t1 seconds
        int t1 = distr(gen);
        this_thread::sleep_for(chrono::seconds(t1));


        //now, a new car arrives
        //first check if the parking lot is not completely full
        //if it is, the car will not wait and get out
        {
            unique_lock<mutex> lock(m);
            if (occupied_spots >= NUM_SPOTS){
                cout << "Parking lot full. Car " << i << " goes away" << endl;
                lock.unlock();
                continue;
            }

            cout << "Car " << i << " parked succesfully." << endl;

            //occupy a spot
            ++occupied_spots;
            //if you are the first car, start the sim
            if (simDuration < 0) {
                simDuration = 0;
                cout << "First car entered, simulation starts now." << endl;
                timeStart = chrono::steady_clock::now();
            } 
            //wake up a threadDeparture
            cv.notify_one();
        }
    }

    //set stop to true
    {   
        lock_guard<mutex> lock(m);
        stop = true;
    }
}


void threadDeparture(){

    random_device rd; 
    mt19937 gen(rd()); //set seed generator
    uniform_int_distribution<> distr(4, 7); //define uniform distributuon for the range [4; 7]

    while (true){
        {
            unique_lock<mutex> lock(m);

            //if (stop) return; //end of sim

            cv.wait(lock, [] {return occupied_spots > 0 || stop; });

        }

        //simulate t2, so the time spent in the parking lot
        int t2 = distr(gen);
        this_thread::sleep_for(chrono::seconds(t2));

        //then leave
        {
            unique_lock<mutex> lock(m);
            if (occupied_spots > 0) {
                --occupied_spots;

                cout << "Car left parking lot" << endl; 
            }

            //check end of sim everytime
            if (stop && occupied_spots == 0) return;
        }

        
    }
}


int main(void){

    //start threads
    thread t0(threadArrival);
    thread t1(threadDeparture);


    t0.join();
    t1.join();

    auto timeFinish = chrono::steady_clock::now();
    cout << "Parking lot emptied. End of simulation" << endl;
    cout << "Total simulation time: " << chrono::duration_cast<std::chrono::seconds>(timeFinish - timeStart).count() << " seconds";


    return 0;
}


