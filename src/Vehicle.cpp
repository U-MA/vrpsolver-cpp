#include "Vehicle.h"

extern "C"
{
#include "vrp_macros.h"
}


Vehicle::Vehicle(void)
{
    routeSize  = 0;
    quantity_  = 0;
    for (int i=0; i < MAXSIZE; i++)
    {
        isVisit_[i] = false;
    }
}

Vehicle::Vehicle(const Vehicle& v)
{
    routeSize  = v.routeSize;
    quantity_  = v.quantity_;
    for (int i=0; i < MAXSIZE; i++)
        isVisit_[i] = v.isVisit_[i];

    for (int i=0; i < routeSize; i++)
        route[i] = v.route[i];
}

Vehicle::~Vehicle(void)
{
}

void Vehicle::init(void)
{
    routeSize  = 0;
    quantity_  = 0;

    for (int i=0; i < MAXSIZE; i++)
        isVisit_[i] = false;
}

Vehicle Vehicle::copy(void) const
{
    Vehicle v_copy;

    for (int i=0; i < MAXSIZE; i++)
        v_copy.isVisit_[i] = isVisit_[i];

    for (int i=0; i < routeSize; i++)
        v_copy.route[i] = route[i];

    v_copy.routeSize = routeSize;
    v_copy.quantity_ = quantity_;
    return v_copy;
}

bool Vehicle::isVisit(int customer) const
{
    return isVisit_[customer-1];
}


/* customerは０以上顧客数未満 */
bool Vehicle::visit(const vrp_problem *vrp, int customer)
{
    if (!(0 < customer && customer < vrp->vertnum))
        return false;

    route[routeSize] = customer;
    isVisit_[customer-1] = true;
    routeSize++;
    quantity_ += vrp->demand[customer];
    return true;
}

int Vehicle::quantity(void) const
{
    return quantity_;
}

int Vehicle::computeCost(const vrp_problem *vrp) const
{
    if (routeSize == 0) return 0;

    int i;
    int cost = vrp->dist.cost[INDEX(0, route[0])];
    for (i=1; i < routeSize; i++)
    {
        cost += vrp->dist.cost[INDEX(route[i-1], route[i])];
    }
    cost += vrp->dist.cost[INDEX(route[i-1], 0)];

    return cost;
}

void Vehicle::print(void) const
{
    printf("[%6d] ", quantity_);
    for (int i=0; i < routeSize; i++)
        printf("%3d", route[i]);
    printf("\n");
}
