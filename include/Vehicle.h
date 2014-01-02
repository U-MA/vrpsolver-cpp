#ifndef VRPSOLVER_CPP_VEHICLE_H
#define VRPSOLVER_CPP_VEHICLE_H

#include <vector>

extern "C"
{
#include "vrp_types.h"
}

#define OUT_OF_BOUND -1

using namespace std;

class Vehicle
{
public:
    Vehicle(void);
    Vehicle(const Vehicle& v);
    ~Vehicle(void);

    void init(void);
    bool empty(void) const;
    bool isVisit(int customer) const;
    bool visit(const vrp_problem *vrp, int customer);
    int  quantity(void) const;
    int  computeCost(const vrp_problem *vrp) const;
    void print(void) const;

private:
    static const int MAXSIZE = 130;
    bool isVisit_[MAXSIZE];
    int  route[MAXSIZE];
    int  routeSize;
    int  quantity_;
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
