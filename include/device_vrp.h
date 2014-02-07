#ifndef VRPSOLVER_CPP_DEVICE_VRP_H
#define VRPSOLVER_CPP_DEVICE_VRP_H

#include "base_vrp.h"
#include "host_vrp.h"

class DeviceVrp : public BaseVrp
{
public:
    DeviceVrp(void);
    DeviceVrp(const HostVrp&);
    ~DeviceVrp(void);

    const DeviceVrp& operator=(const HostVrp&);

    int customer_size(void)  const { return vrp_->vertnum-1; }
    int vehicle_size(void)   const { return vrp_->numroutes; }
    int capacity(void)       const { return vrp_->capacity; }
    int cost(int v0, int v1) const { return vrp_->dist.cost[INDEX(v0, v1)]; }
    int demand(int v)        const { return vrp_->demand[v]; }

private:
    vrp_problem *vrp_;
};

#endif /* VRPSOLVER_CPP_DEVICE_VRP_H */
