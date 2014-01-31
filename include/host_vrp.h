#ifndef VRPSOLVER_CPP_HOST_VRP_H
#define VRPSOLVER_CPP_HOST_VRP_H

extern "C"
{
#include "vrp_macros.h"
#include "vrp_types.h"
}

#include "base_vrp.h"

class HostVrp : public BaseVrp
{
public:
    HostVrp(void);
    HostVrp(const char *file_path);
    ~HostVrp(void);

    void Create(const char *file_path);

    const char* name(void)        const;
    int customer_size(void)  const { return vrp_->vertnum-1; }
    int vehicle_size(void)   const { return vrp_->numroutes; }
    int capacity(void)       const { return vrp_->capacity; }
    int cost(int v0, int v1) const { return vrp_->dist.cost[INDEX(v0, v1)]; }
    int demand(int v)        const { return vrp_->demand[v]; }

private:
    vrp_problem *vrp_;
};

#endif /* VRPSOLVER_CPP_HOST_VRP_H */
