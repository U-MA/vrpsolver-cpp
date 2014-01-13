#ifndef VRPSOLVER_CPP_VEHICLE_H
#define VRPSOLVER_CPP_VEHICLE_H

extern "C"
{
#include "vrp_types.h"
}

#define OUT_OF_BOUND -1

class Vehicle
{
public:
    Vehicle(void);
    ~Vehicle(void);

    Vehicle copy(void) const;

    /* getter */
    int  quantity(void) const;

    bool visit(const vrp_problem *vrp, int customer);
    int  computeCost(const vrp_problem *vrp) const;

    void print(void) const;

private:
    static const int MAXSIZE = 130;

    int  route_[MAXSIZE];
    int  route_length_;
    int  quantity_;
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
