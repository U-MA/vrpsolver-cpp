#include <stdio.h>
#include <string.h>

#include "Vehicle.h"

extern "C"
{
#include "vrp_macros.h"
}


Vehicle Vehicle::copy(void) const
{
    Vehicle v_copy;

    /* 使っている部分だけコピー */
    const size_t route_bytes = route_length_ * sizeof(int);
    memcpy(v_copy.route_, route_, route_bytes);

    v_copy.route_length_ = route_length_;
    v_copy.capacity_     = capacity_;

    return v_copy;
}

int Vehicle::capacity(void) const
{
    return capacity_;
}

bool Vehicle::isOverCapacity(const vrp_problem *vrp, int customer) const
{
    return (capacity_ + vrp->demand[customer] > vrp->capacity);
}

/* 0はdepotを表すため、範囲外 */
__host__ __device__
static bool customerIsInBound(int customer, int customer_end)
{
    return (0 < customer && customer < customer_end);
}

/* customerは０以上顧客数未満 */
bool Vehicle::visit(const vrp_problem *vrp, int customer)
{
    if (!customerIsInBound(customer, vrp->vertnum))
        return false;

    route_[route_length_++] = customer;
    capacity_ += vrp->demand[customer];
    return true;
}

__host__ __device__
static int INDEX_CU(int v0, int v1)
{
    return( (v1) > (v0) ? ((int)(v1))*((v1)-1)/2+(v0) :
                          ((int)(v0))*((v0)-1)/2+(v1));
}


int Vehicle::computeCost(const vrp_problem *vrp) const
{
    if (route_length_ == 0) return 0;

    int i;
    int cost = vrp->dist.cost[INDEX_CU(0, route_[0])];
    for (i=1; i < route_length_; i++)
    {
        cost += vrp->dist.cost[INDEX_CU(route_[i-1], route_[i])];
    }
    cost += vrp->dist.cost[INDEX_CU(route_[i-1], 0)];

    return cost;
}

void Vehicle::print(void) const
{
    printf("[%6d] ", capacity_);
    for (int i=0; i < route_length_; i++)
        printf("%4d", route_[i]);
    printf("\n");
}
