#include "Vehicle.h"

extern "C"
{
#include "vrp_macros.h"
}


Vehicle::Vehicle(void)
{
}

Vehicle::Vehicle(int customerSize)
{
    routeSize = 0;
    quantity  = 0;
}

Vehicle::Vehicle(const Vehicle& v)
{
}

Vehicle::~Vehicle(void)
{
}

/* customerは０以上顧客数未満 */
bool Vehicle::visit(vrp_problem *vrp, int customer)
{
    if (!(0 < customer && customer < vrp->vertnum))
        return false;

    route[routeSize] = customer;
    routeSize++;
    quantity += vrp->demand[customer];
    return true;
}

int Vehicle::getQuantity(void)
{
    return quantity;
}

int Vehicle::getRoute(int idx)
{
    if (routeSize == 0 || routeSize < idx) return OUT_OF_BOUND;
    return route[idx];
}

int Vehicle::computeCost(vrp_problem *vrp)
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

void Vehicle::print(void)
{
    printf("[%6d] ", quantity);
    for (int i=0; i < routeSize; i++)
        printf("%3d", route[i]);
    printf("\n");
}
