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

    bool visit(vrp_problem *vrp, int customer);

    int getQuantity(void);
    int getRoute(int idx); /* 早期に実装しすぎた */

    int computeCost(vrp_problem *vrp);

    void print(void);
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
