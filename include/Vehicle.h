#ifndef VRPSOLVER_CPP_VEHICLE_H
#define VRPSOLVER_CPP_VEHICLE_H

extern "C"
{
#include "vrp_types.h"
}

class Vehicle
{
public:
    Vehicle(void) : route_(), route_length_(0), capacity_(0) {};

    int  capacity(void) const;

    void visit(const vrp_problem *vrp, int customer);
    int  computeCost(const vrp_problem *vrp) const;

    void print(void) const;

private:
    static const int kMaxSize = 130;

    int route_[kMaxSize];
    int route_length_;
    int capacity_;
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
