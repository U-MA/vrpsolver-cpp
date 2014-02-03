#ifndef VRPSOLVER_CPP_DEVICE_SIMULATOR_H
#define VRPSOLVER_CPP_DEVICE_SIMULATOR_H

#include "host_vrp.h"
#include "vehicle_manager.h"

class DeviceSimulator
{
public:
    DeviceSimulator() {}

    int Run(const HostVrp& host_vrp, const VehicleManager& vm, int count);

private:
};

#endif /* VRPSOLVER_CPP_DEVICE_SIMULATOR_H */
