#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#define NUM_SPOTS 3

using namespace std;


mutex m;
int simDuration = -1; //starting from firs car that enters
int freeSpots = NUM_SPOTS;
int simDuration_unscaled = -1;
int timeStart = -1;


void car(){

    random_device rd; 
    mt19937 gen(rd()); //set seed generator
    uniform_int_distribution<> distr1(1, 3); //define uniform distributuon for the range [1; 3]
    uniform_int_distribution<> distr2(4, 7); //define uniform distribution for the range [4; 7]


    //wait for tot seconds, then arrive at the parking lot
    int t1 = distr1(gen);
    this_thread::sleep_for(chrono::seconds(t1));

    //try to look for a spot
    {
        lock_guard<mutex> lock(m);
        if (freeSpots > 0){
            --freeSpots;
            //the first car starts sim
            if (simDuration == -1){
                simDuration = 0;
                simDuration_unscaled = t1;
                timeStart = t1;
            }
        } else {
            return; //leave parking lot if full
        }
    }

    //stay parked for tot seconds
    int t2 = distr2(gen);
    this_thread::sleep_for(chrono::seconds(t2));

    //leave the spot
    {
        lock_guard<mutex> lock(m);
        ++freeSpots;

        //update sim
        if (simDuration == 0){
            //first car count just t2
            simDuration += t2;
            simDuration_unscaled += t2;
        } else {
            //other cars
            int totTime = t1+t2;
            if (totTime > simDuration_unscaled){
                simDuration_unscaled += (totTime - simDuration_unscaled);
                simDuration = simDuration_unscaled - timeStart;
            }
        }
    }
}


