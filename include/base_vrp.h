#ifndef VRPSOLVER_CPP_BASE_VRP_H
#define VRPSOLVER_CPP_BASE_VRP_H

extern "C"
{
#include "vrp_macros.h"
#include "vrp_types.h"
}

class BaseVrp
{
public:
    BaseVrp(void) : vrp_(NULL) {};
    virtual ~BaseVrp(void) {};

    int customer_size(void) const  { return vrp_->vertnum-1; }
    int vehicle_size(void) const   { return vrp_->numroutes; }
    int capacity(void) const       { return vrp_->capacity; }
    int cost(int v0, int v1) const { return vrp_->dist.cost[INDEX(v0, v1)]; }
    int demand(int v) const        { return vrp_->demand[v]; }

protected:
    vrp_problem *vrp_;
};

#endif /* VRPSOLVER_CPP_BASE_VRP_H */
