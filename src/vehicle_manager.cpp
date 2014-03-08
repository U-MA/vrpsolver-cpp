#include <string.h>
#include <stdio.h>

#include "vehicle_manager.h" 

bool VehicleManager::isVisit(int customer) const
{
    return is_visit_[customer-1];
}

bool VehicleManager::isVisitAll(const BaseVrp& vrp) const
{
    const int customer_size = vrp.CustomerSize();
    for (int i=1; i <= customer_size; i++)
        if (!isVisit(i)) return false;

    return true;
}

bool VehicleManager::canVisitCustomer(const BaseVrp& vrp, int customer) const
{
    return vehicle_[vehicle_size_-1].Capacity() + vrp.Demand(customer) <=
           vrp.Capacity();
}

bool VehicleManager::nextVehicleRemain(const BaseVrp& vrp) const
{
    return ((unsigned)vehicle_size_ < vrp.VehicleSize());
}

void VehicleManager::move(const BaseVrp& vrp, int customer)
{
    /* move内でchangeVehicle()しているが将来取り除く */
    bool is_change_vehicle = (customer == kChange);
    if (is_change_vehicle)
        changeVehicle();
    else
    {
        vehicle_[vehicle_size_-1].Visit(vrp, customer);
        is_visit_[customer-1] = true;
    }
}

bool VehicleManager::isMovable(const BaseVrp& vrp) const
{
    int vertnum = vrp.CustomerSize() + 1;
    for (int i=1; i < vertnum; i++)
        if (!isVisit(i) && canVisitCustomer(vrp, i))
            return true;

    if (nextVehicleRemain(vrp)) return true;

    return false;
}

int VehicleManager::computeTotalCost(const BaseVrp& vrp) const
{
    int total_cost = 0;
    for (int i=0; i < vehicle_size_; i++)
        total_cost += vehicle_[i].ComputeCost(vrp);

    return total_cost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < vehicle_size_; i++)
    {
        printf("vehicle %2d", i);
        vehicle_[i].Print();
    }
}
