#include "device_simulator.h"
#include "device_vrp.h"


__global__
void simulation_kernel(DeviceVrp *, VehicleManager *, int *);

DeviceSimulator::Run(const HostVrp& host_vrp, const VehicleManager& vm, int count)
{
    DeviceVrp device_vrp(host_vrp);

    thrust::device_vector<VehicleManager> device_vms(count);

    /* vmの内容をdevice_vmsにコピー */
    thrust::generate(device_vms.begin(), device_vms.end(), vm);

    thrust::device_vector<int> device_costs(count);

    VehicleManager *device_vms_ptr = thrust::raw_pointer_cast(device_vms);
    int *device_costs_ptr = thrust::raw_pointer_cast(device_costs);

    const int vertnum = host_vrp->customer_size()+1;
    simulation_kernel<<<count, vertnum, vertnum*sizeof(int)>>>(
            device_vrp, device_vms_ptr, device_costs_ptr);

    return thrust::reduce(device_costs_ptr.begin(),
                          device_costs_ptr.end(),
                          thrust::min_element());
}


__global__ void simulation_kernel(DeviceVrp *device_vrp, VehicleManager *device_vms,
                                  int *device_costs)
{
}
