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
};

#endif /* VRPSOLVER_CPP_DEVICE_VRP_H */
