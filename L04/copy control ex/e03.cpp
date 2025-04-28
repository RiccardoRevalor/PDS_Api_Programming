#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

class C {
private:
    int i;
public:
    C () {
        cout << "[C]";
    }
    ~C() {
        cout << "[D]";
    }
    C (const C &n) {
        cout << "[CC]";
    }
    C &operator=(const C &n) {
        cout << "[CAO]";
        return *this;
    }
    C (C&& n) noexcept {
        cout << "[MC]";
    }
    C &operator=(C&&n) noexcept {
        cout << "[MAO]";
        return *this;
    }
    void set(int n)  {
        i = n;
    };
    int get () {
        return i;
    }
};

void swap (C &e1, C &e2) {
  C tmp;		//constructor 1 volta
  tmp=e1;		//copy assignment operator 1 volta
  e1=e2;		//copy assignmento perator 1 volta
  e2=tmp;		//copy assignment operator 1 volta
  return;		//destructor di tmp 
}

int main() {
  cout << endl << "{01}"; C e1;  //constructor 1 volta	
  cout << endl << "{02}"; C e2[5];	//constructor 5 volte
  cout << endl << "{03}"; C e3 = *new (std::nothrow) C;	//constructor (per new (std::nothrow) C), copy constructor (per C e3)
  cout << endl << "{04}"; C *e4 = new C;		//constructor (per new C) e basta siccome e4 è solo un pointer
  cout << endl << "{05}"; C *e5 = new C[10];  	//constructor (per new C) 10 volte

  
  cout << endl << "{06}"; C v1 = e1;  //copy constructor 1 volta
  cout << endl << "{07}"; C v2 = (std::move(e1));	//move constrctor 1 volta
  cout << endl << "{08}"; C v3;		//constructor 1 volta
  cout << endl << "{09}"; v3 = (std::move(e1));	//move assignment operator 1 volta

  cout << endl << "{10}"; swap (e1, v3);		//4 

  cout << endl << "{11}"; return 0;                          //destructor di oggetti ( e non pointers): e1, e2[5], e3, new C, new C[10], v1, v2, v3 -> 1+5+1+10+1+1+1=
				              //new (std::nothrow) C is a pointer! e4,e5 are pointers -> no destructors (memory leak since delete is not called...)
}