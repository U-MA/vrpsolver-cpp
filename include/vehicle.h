#ifndef VRPSOLVER_CPP_VEHICLE_H
#define VRPSOLVER_CPP_VEHICLE_H

#include "base_vrp.h"

class Vehicle
{
public:
    Vehicle(void) : route_(), route_length_(0), capacity_(0), is_visit_()
    {
        for (int i=0; i < kMaxSize; i++)
            is_visit_[i] = false;
    };

    int  capacity(void) const { return capacity_; }
    bool is_visit(int customer_id) const { return is_visit_[customer_id-1]; }

    void visit(const BaseVrp& vrp, int customer);
    int  computeCost(const BaseVrp& vrp) const;

    void print(void) const;

private:
    static const int kMaxSize = 130;

    int  route_[kMaxSize];
    int  route_length_;
    int  capacity_;
    bool is_visit_[kMaxSize];
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
