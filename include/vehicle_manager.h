#ifndef VRPSOLVER_CPP_VEHICLE_MANAGER_H
#define VRPSOLVER_CPP_VEHICLE_MANAGER_H

extern "C"
{
#include "vrp_types.h"
}

#include "vehicle.h"

class VehicleManager
{
public:
    static const int kChange = 0;

    VehicleManager(void) : vehicle_(), vehicle_size_(1), is_visit_() {};

    VehicleManager copy(void) const;

    /* getter */
    int  vehicle_size(void) const;

    bool isVisit(int customer) const;
    bool isVisitAll(const vrp_problem *vrp) const;

    bool isFinish(const vrp_problem *vrp) const;
    bool checkCapacityConstraint(const vrp_problem *vrp, int customer) const;
    bool canVisit(const vrp_problem *vrp, int customer) const;
    bool nextVehicleRemain(const vrp_problem *vrp) const;

    bool move(const vrp_problem *vrp, int move);
    int  computeTotalCost(const vrp_problem *vrp) const;

    void print(void) const;
   
private:
    static const int kVehicleMax  = 20;
    static const int kCustomerMax = 200;

    bool changeVehicle(const vrp_problem *vrp);

    Vehicle vehicle_[kVehicleMax];
    int     vehicle_size_;
    bool    is_visit_[kCustomerMax];
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
