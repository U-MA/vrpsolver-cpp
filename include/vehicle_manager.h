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

    VehicleManager(void) : vehicle_(), is_visit_(), size_(1) {};

    VehicleManager copy(void) const;

    /* getter */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int  size(void) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool isVisit(int customer) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool isVisitAll(const vrp_problem *vrp) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool isFinish(const vrp_problem *vrp) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool canVisit(const vrp_problem *vrp, int customer) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool nextVehicleRemain(const vrp_problem *vrp) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool move(const vrp_problem *vrp, int move);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int  computeTotalCost(const vrp_problem *vrp) const;

#ifdef __CUDACC__
    __host__ __device__
#endif
    void print(void) const;
   
private:
    static const int kVehicleMax  = 20;
    static const int kCustomerMax = 200;

#ifdef __CUDACC__
    __host__ __device__
#endif
    bool changeVehicle(const vrp_problem *vrp);

    Vehicle vehicle_[kVehicleMax];
    bool    is_visit_[kCustomerMax];
    int     size_; /* 車体の数 */
};

#endif /* VRPSOLVER_CPP_VEHICLE_MANAGER_H */
