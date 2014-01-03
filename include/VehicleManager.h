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
    bool    update(const vrp_problem *vrp, int customer);
    bool    isVisit(int customer) const;
    bool    isVisitAll(const vrp_problem *vrp) const;

    VehicleManager    copy(void) const;

    /* accessor */
    Vehicle getVehicle(int id);
    Vehicle getVehicle(void);
    int     size(void) const;

    void    add(Vehicle& v);
    int     computeTotalCost(const vrp_problem *vrp) const;

    void    print(void) const;
   
private:
    static const int VEHICLE_MAX  = 20;
    static const int CUSTOMER_MAX = 200;
    Vehicle vehicle[VEHICLE_MAX];
    bool    isVisit_[CUSTOMER_MAX];
    int     runVehicle; /* 現在走行している車体 */
    int     size_;      /* vehicleに含まれてるVehicleの数 */
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
