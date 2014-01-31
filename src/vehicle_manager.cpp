#include <string.h>
#include <stdio.h>

#include "vehicle_manager.h" 

bool VehicleManager::isVisit(int customer) const
{
    return is_visit_[customer-1];
}

/* DEPRECATED */
bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    const int customer_size = vrp->vertnum-1;
    for (int i=1; i <= customer_size; i++)
        if (!isVisit(i)) return false;

    return true;
} /* DEPRECATED */

bool VehicleManager::isVisitAll(const BaseVrp& vrp) const
{
    const int customer_size = vrp.customer_size();
    for (int i=1; i <= customer_size; i++)
        if (!isVisit(i)) return false;

    return true;
}

/* DEPRECATED */
bool VehicleManager::canVisitCustomer(const vrp_problem *vrp, int customer) const
{
    return vehicle_[vehicle_size_-1].capacity() + vrp->demand[customer] <=
           vrp->capacity;
} /* DEPRECATED */

bool VehicleManager::canVisitCustomer(const BaseVrp& vrp, int customer) const
{
    return vehicle_[vehicle_size_-1].capacity() + vrp.demand(customer) <=
           vrp.capacity();
}

/* DEPRECATED */
bool VehicleManager::nextVehicleRemain(const vrp_problem *vrp) const
{
    return (vehicle_size_ < vrp->numroutes);
} /* DEPRECATED */

bool VehicleManager::nextVehicleRemain(const BaseVrp& vrp) const
{
    return (vehicle_size_ < vrp.vehicle_size());
}

/* DEPRECATED */
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
} /* DEPRECATED */

void VehicleManager::move(const BaseVrp& vrp, int customer)
{
    /* move内でchangeVehicle()しているが将来取り除く */
    bool is_change_vehicle = (customer == kChange);
    if (is_change_vehicle)
        changeVehicle();
    else
    {
        vehicle_[vehicle_size_-1].visit(vrp, customer);
        is_visit_[customer-1] = true;
    }
}

/* DEPRECATED */
bool VehicleManager::isMovable(const vrp_problem *vrp) const
{
    for (int i=1; i < vrp->vertnum; i++)
        if (!isVisit(i) && canVisitCustomer(vrp, i))
            return true;

    if (nextVehicleRemain(vrp)) return true;

    return false;
} /* DEPRECATED */

bool VehicleManager::isMovable(const BaseVrp& vrp) const
{
    int vertnum = vrp.customer_size() + 1;
    for (int i=1; i < vertnum; i++)
        if (!isVisit(i) && canVisitCustomer(vrp, i))
            return true;

    if (nextVehicleRemain(vrp)) return true;

    return false;
}


/* DEPRECATED */
int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int total_cost = 0;
    for (int i=0; i < vehicle_size_; i++)
        total_cost += vehicle_[i].computeCost(vrp);

    return total_cost;
} /* DEPRECATED */

int VehicleManager::computeTotalCost(const BaseVrp& vrp) const
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
