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
    int vehicleSize;
    Vehicle vehicle[VEHICLE_MAX];
    bool isVisit[CUSTOMER_MAX];

public:

    VehicleManager(void);
    VehicleManager(int vehicleSize);
    ~VehicleManager(void);

    int getRunningVehicleNumber(void);

    bool changeVehicle(void);
    bool update(vrp_problem *vrp, int customer);

    bool isVisitAll(vrp_problem *vrp);

    int computeTotalCost(vrp_problem *vrp);

    int randomSimulation(vrp_problem *vrp);

    void print(void);
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
