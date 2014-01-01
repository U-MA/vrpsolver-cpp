#include "VehicleManager.h"


VehicleManager::VehicleManager(void)
{
    runVehicle = 0;
    for (int i=0; i < CUSTOMER_MAX; i++)
        isVisit[i] = false;
}


VehicleManager::~VehicleManager(void)
{
}

int VehicleManager::getSize(void) const
{
    return vehicle_vec.size();
}

void VehicleManager::add(Vehicle& v)
{
    vehicle_vec.push_back(v);
}

Vehicle VehicleManager::getVehicle(int id)
{
    return vehicle_vec[id];
}

Vehicle VehicleManager::getVehicle(void)
{
    return vehicle_vec[0];
}

bool VehicleManager::empty(void)
{
    return vehicle_vec.empty();
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
    int size = vehicle_vec.size();
    for (int i=0; i < size; i++)
    {
        if (vehicle_vec[i].isVisitOne(customer))
            return true;
    }
    return false;
}

bool VehicleManager::changeVehicle(void)
{
    if (vehicle_vec.size() <= (unsigned)runVehicle+1) return false;

    runVehicle++;
    return true;
}

bool VehicleManager::update(const vrp_problem *vrp, int customer)
{
    if (vehicle_vec[runVehicle].visit(vrp, customer))
        return (isVisit[customer-1] = true);

    return false;
}

int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int totalCost = 0;
    int size = vehicle_vec.size();
    for (int i=0; i < size; i++)
        totalCost += vehicle_vec[i].computeCost(vrp);

    return totalCost;
}

void VehicleManager::print(void) const
{
    int size = vehicle_vec.size();
    for (int i=0; i < size; i++)
    {
        printf("vehicle %2d", i);
        vehicle[i].print();
    }
}
