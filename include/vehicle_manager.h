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

    /* ACCESSOR */
    int  vehicle_size(void) const { return vehicle_size_; }
    void changeVehicle(void) { vehicle_size_++; }

    bool isVisit(int customer) const;
    bool isVisitAll(const vrp_problem *vrp) const;

    /* 次の手があるかどうか */
    bool isMovable(const vrp_problem *vrp) const;

    /* 現在走行している車体はcustomerを訪問出来るか */
    bool canVisitCustomer(const vrp_problem *vrp, int customer) const;

    /* 次の車体があるかどうか */
    bool nextVehicleRemain(const vrp_problem *vrp) const;

    /* VehicleMangerが次の手としてmoveを行う */
    void move(const vrp_problem *vrp, int move);

    /* VehicleManagerが管理している車体のコストの和を返す */
    int  computeTotalCost(const vrp_problem *vrp) const;

    /* VehicleManagerが管理している車体のルートを出力 */
    void print(void) const;

private:
    static const int kVehicleMax  = 20;
    static const int kCustomerMax = 200;


    Vehicle vehicle_[kVehicleMax];
    int     vehicle_size_;
    bool    is_visit_[kCustomerMax];
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
