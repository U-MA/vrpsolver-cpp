#include "VehicleManager.h"


VehicleManager::VehicleManager(void)
{
    runVehicle = 0;
    size_ = 0;
    for (int i=0; i < CUSTOMER_MAX; i++)
        isVisit[i] = false;
}

VehicleManager::~VehicleManager(void)
{
}

int VehicleManager::size(void) const
{
    return size_;
}

void VehicleManager::add(Vehicle& v)
{
    vehicle[size_] = v;
    size_++;
}

Vehicle VehicleManager::getVehicle(int id)
{
    return vehicle[id];
}

Vehicle VehicleManager::getVehicle(void)
{
    return vehicle[0];
}

bool VehicleManager::empty(void)
{
    return size_ == 0;
}

bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    int customerSize = vrp->vertnum-1;
    for (int i=1; i < customerSize; i++)
        if (!isVisitOne(i)) return false;

    return true;
}

bool VehicleManager::isVisitOne(int customer) const
{
    for (int i=0; i < size_; i++)
    {
        if (vehicle[i].isVisitOne(customer))
            return true;
    }
    return false;
}

bool VehicleManager::changeVehicle(void)
{
    if (size_ <= runVehicle+1) return false;

    runVehicle++;
    return true;
}

bool VehicleManager::update(const vrp_problem *vrp, int customer)
{
    if (vehicle[runVehicle].visit(vrp, customer))
        return (isVisit[customer-1] = true);

    return false;
}

int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int totalCost = 0;
    for (int i=0; i < size_; i++)
        totalCost += vehicle[i].computeCost(vrp);

    return totalCost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < size_; i++)
    {
        printf("vehicle %2d", i);
        vehicle[i].print();
    }
}
