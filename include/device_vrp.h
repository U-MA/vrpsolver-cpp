#ifndef VRPSOLVER_CPP_DEVICE_VRP_H
#define VRPSOLVER_CPP_DEVICE_VRP_H

#include "base_vrp.h"
#include "host_vrp.h"

class DeviceVrp : public BaseVrp
{
public:
    DeviceVrp();
    DeviceVrp(const HostVrp&);
    ~DeviceVrp();

    const DeviceVrp& operator=(const HostVrp&);

    unsigned int CustomerSize()  const { return vrp_->vertnum-1; }
    unsigned int VehicleSize()   const { return vrp_->numroutes; }
    unsigned int Capacity()       const { return vrp_->capacity; }
    unsigned int Cost(int v0, int v1) const { return vrp_->dist.cost[INDEX(v0, v1)]; }
    unsigned int Demand(int v)        const { return vrp_->demand[v]; }
};

#endif /* VRPSOLVER_CPP_DEVICE_VRP_H */
