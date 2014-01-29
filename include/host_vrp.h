#ifndef VRPSOLVER_CPP_HOST_VRP_H
#define VRPSOLVER_CPP_HOST_VRP_H

#include "base_vrp.h"

class HostVrp : public BaseVrp
{
public:
    HostVrp(void);
    HostVrp(const char *file_path);
    ~HostVrp(void);

    void Create(const char *file_path);
};

#endif /* VRPSOLVER_CPP_HOST_VRP_H */
