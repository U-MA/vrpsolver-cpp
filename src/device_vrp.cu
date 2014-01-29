#include "device_vrp.h"

DeviceVrp::DeviceVrp(void)
{
    cudaMalloc((void **)&vrp_, sizeof(vrp_problem));
}

DeviceVrp::DeviceVrp(const HostVrp& host_vrp)
{
    /* TODO */
}

DeviceVrp::~DeviceVrp(void)
{
    /* TODO */
}

const DeviceVrp& operator=(const HostVrp& host_vrp)
{
    /* TODO */
}
