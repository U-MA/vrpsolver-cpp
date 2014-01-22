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

    Vehicle copy(void) const;

    /* getter */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int  capacity(void) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool isOverCapacity(const vrp_problem *vrp, int customer) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool visit(const vrp_problem *vrp, int customer);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int  computeCost(const vrp_problem *vrp) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    void print(void) const;

private:
    static const int kMaxSize = 130;

    int route_[kMaxSize];
    int route_length_;
    int capacity_;
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
