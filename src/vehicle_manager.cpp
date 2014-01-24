#include <string.h>
#include <stdio.h>

#include "vehicle_manager.h" 

bool VehicleManager::isVisit(int customer) const
{
    return is_visit_[customer-1];
}

bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    const int customer_size = vrp->vertnum-1;
    for (int i=1; i <= customer_size; i++)
        if (!isVisit(i)) return false;

    return true;
}

bool VehicleManager::isInCapacityConstraint(const vrp_problem *vrp, int customer) const
{
    return vehicle_[vehicle_size_-1].capacity() + vrp->demand[customer] <=
           vrp->capacity;
}

bool VehicleManager::nextVehicleRemain(const vrp_problem *vrp) const
{
    return (vehicle_size_ < vrp->numroutes);
}

void VehicleManager::move(const vrp_problem *vrp, int move)
{
    bool is_change_vehicle = (move == kChange);
    if (is_change_vehicle)
        changeVehicle();
    else
    {
        vehicle_[vehicle_size_-1].visit(vrp, move);
        is_visit_[move-1] = true;
    }
}

bool VehicleManager::isMovable(const vrp_problem *vrp) const
{
    for (int i=1; i < vrp->vertnum; i++)
        if (!isVisit(i) && isInCapacityConstraint(vrp, i))
            return true;

    if (nextVehicleRemain(vrp)) return true;

    return false;
}


int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int total_cost = 0;
    for (int i=0; i < vehicle_size_; i++)
        total_cost += vehicle_[i].computeCost(vrp);

    return total_cost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < vehicle_size_; i++)
    {
        printf("vehicle %2d", i);
        vehicle_[i].print();
    }
}
