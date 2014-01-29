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
};

#endif /* VRPSOLVER_CPP_HOST_VRP_H */
