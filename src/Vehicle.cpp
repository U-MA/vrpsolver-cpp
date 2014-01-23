#include <stdio.h>
#include <string.h>

#include "Vehicle.h"

extern "C"
{
#include "vrp_macros.h"
}


int Vehicle::capacity(void) const
{
    return capacity_;
}

bool Vehicle::isOverCapacity(const vrp_problem *vrp, int customer) const
{
    return (capacity_ + vrp->demand[customer] > vrp->capacity);
}

void Vehicle::visit(const vrp_problem *vrp, int customer)
{
    route_[route_length_++] = customer;
    capacity_ += vrp->demand[customer];
}

int Vehicle::computeCost(const vrp_problem *vrp) const
{
    if (route_length_ == 0) return 0;

    int i, cost = vrp->dist.cost[INDEX(0, route_[0])];
    for (i=1; i < route_length_; i++)
    {
        cost += vrp->dist.cost[INDEX(route_[i-1], route_[i])];
    }
    cost += vrp->dist.cost[INDEX(route_[i-1], 0)];

    return cost;
}

void Vehicle::print(void) const
{
    printf("[%6d] ", capacity_);
    for (int i=0; i < route_length_; i++)
        printf("%4d", route_[i]);
    printf("\n");
}
