#ifndef VRPSOLVER_CPP_VEHICLE_MANAGER_H
#define VRPSOLVER_CPP_VEHICLE_MANAGER_H

#include "base_vrp.h"
#include "vehicle.h"

class VehicleManager
{
public:
    static const int kChange = 0;

    VehicleManager(void) : vehicle_(), vehicle_size_(1), is_visit_() {};

    int  vehicle_size(void) const { return vehicle_size_; }
    void changeVehicle(void) { vehicle_size_++; }

    bool isVisit(int customer) const;
    bool isVisitAll(const BaseVrp& vrp) const;

    bool isMovable(const BaseVrp& vrp) const;
    bool canVisitCustomer(const BaseVrp& vrp, int customer) const;

    bool nextVehicleRemain(const BaseVrp& vrp) const;

    void move(const BaseVrp& vrp, int customer);

    int  computeTotalCost(const BaseVrp& vrp) const;

    void print(void) const;

private:
    static const int kVehicleMax  = 20;
    static const int kCustomerMax = 200;


    Vehicle vehicle_[kVehicleMax];
    int     vehicle_size_;
    bool    is_visit_[kCustomerMax];
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
