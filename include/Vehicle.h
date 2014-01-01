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
private:
    static const int MAXSIZE = 130;
    int route[MAXSIZE];
    int routeSize;
    int quantity;

public:
    Vehicle(void);
    Vehicle(int customerSize);
    Vehicle(const Vehicle& v);
    ~Vehicle(void);

    bool empty(void) const;

    bool visit(const vrp_problem *vrp, int customer);

    int getQuantity(void) const;
    int getRoute(int idx) const; /* 早期に実装しすぎた */

    int computeCost(const vrp_problem *vrp) const;

    void print(void) const;
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
