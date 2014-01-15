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
    static const int kChange = 0;

    VehicleManager(void) : vehicle_(), is_visit_(), size_(1) {};

    VehicleManager copy(void) const;

    /* getter */
    int  size(void) const;

    bool isVisit(int customer) const;
    bool isVisitAll(const vrp_problem *vrp) const;
    bool isFinish(const vrp_problem *vrp) const;
    bool canVisit(const vrp_problem *vrp, int customer) const;

    bool move(const vrp_problem *vrp, int move);
    int  computeTotalCost(const vrp_problem *vrp) const;

    void print(void) const;
   
private:
    static const int kVehicleMax  = 20;
    static const int kCustomerMax = 200;

    Vehicle vehicle_[kVehicleMax];
    bool    is_visit_[kCustomerMax];
    int     size_; /* 車体の数 */
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
