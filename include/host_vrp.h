#ifndef VRPSOLVER_CPP_HOST_VRP_H
#define VRPSOLVER_CPP_HOST_VRP_H

extern "C"
{
#include "vrp_macros.h"
#include "vrp_types.h"
}

#include "base_vrp.h"

class HostVrp : BaseVrp
{
public:
    HostVrp(void);
    HostVrp(const char *file_path);
    ~HostVrp(void);

    void Create(const char *file_path);

    int customer_size(void)  const { return vrp_->vertnum-1; }
    int vehicle_size(void)   const { return vrp_->numroutes; }
    int capacity(void)       const { return vrp_->capacity;  }
    int cost(int v1, int v2) const { return vrp_->dist.cost[INDEX(v1, v2)]; }
    int demand(int v) const        { return vrp_->demand[v]; }
};

#endif /* VRPSOLVER_CPP_HOST_VRP_H */
