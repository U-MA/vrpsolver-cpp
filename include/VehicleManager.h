#ifndef VRPSOLVER_CPP_VEHICLE_MANAGER_H
#define VRPSOLVER_CPP_VEHICLE_MANAGER_H

extern "C"
{
#include "vrp_types.h"
}

#include "Vehicle.h"

class VehicleManager
{
public:
    VehicleManager(void);
    ~VehicleManager(void);

    bool    empty(void);
    bool    changeVehicle(void);
    bool    update(const vrp_problem *vrp, int customer);
    bool    isVisitAll(const vrp_problem *vrp) const;
    bool    isVisitOne(int customer) const;
    Vehicle getVehicle(int id);
    Vehicle getVehicle(void);
    int     size(void) const;
    int     computeTotalCost(const vrp_problem *vrp) const;
    void    add(Vehicle& v);
    void    print(void) const;
   
private:
    static const int VEHICLE_MAX  = 20;
    static const int CUSTOMER_MAX = 200;
    Vehicle vehicle[VEHICLE_MAX];
    bool    isVisit[CUSTOMER_MAX];
    int     runVehicle; /* 現在走行している車体 */
    int     size_;      /* vehicleに含まれてるVehicleの数 */
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
