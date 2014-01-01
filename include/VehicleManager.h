#ifndef VRPSOLVER_CPP_VEHICLE_MANAGER_H
#define VRPSOLVER_CPP_VEHICLE_MANAGER_H

extern "C"
{
#include "vrp_types.h"
}

#include "Vehicle.h"

class VehicleManager
{
private:
    static const int VEHICLE_MAX = 20;
    static const int CUSTOMER_MAX = 200;
    int runVehicle; /* 現在走行している車体 */
    int size;
    Vehicle vehicle[VEHICLE_MAX];
    bool isVisit[CUSTOMER_MAX];

public:

    VehicleManager(void);
    VehicleManager(int vehicleSize);
    ~VehicleManager(void);

    int getRunningVehicleNumber(void) const;
    int getEmptyVehicle(void) const;
    Vehicle getVehicle(int id);

    bool empty(void);
    bool changeVehicle(void);
    bool update(const vrp_problem *vrp, int customer);

    bool isVisitAll(const vrp_problem *vrp) const;
    bool isVisitOne(int customer) const;

    int computeTotalCost(const vrp_problem *vrp) const;

    int randomSimulation(const vrp_problem *vrp);

    void print(void) const;
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
