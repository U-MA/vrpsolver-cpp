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
        isVisit[i] = false;
    }
}

Vehicle::Vehicle(const Vehicle& v)
{
    routeSize  = v.routeSize;
    quantity_  = v.quantity_;
    for (int i=0; i < MAXSIZE; i++)
        isVisit[i] = v.isVisit[i];

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
        isVisit[i] = false;
}

bool Vehicle::empty(void) const
{
    return (routeSize == 0);
}

bool Vehicle::isVisitOne(int customer) const
{
    return isVisit[customer-1];
}


/* customerは０以上顧客数未満 */
bool Vehicle::visit(const vrp_problem *vrp, int customer)
{
    if (!(0 < customer && customer < vrp->vertnum))
        return false;

    route[routeSize] = customer;
    isVisit[customer-1] = true;
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
