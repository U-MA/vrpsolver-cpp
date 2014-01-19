#include <stdio.h>

#include "VehicleManager.h"

__global__
void check(int expect, VehicleManager *device_vm, bool *err)
{
    if (expect == device_vm->size())
        *err = false;
    else
        *err = true;
}


/* DRAM上にVehicleManagerを確保するためのテスト */
int main(int argc, char **argv)
{
    VehicleManager host_vehicle_manager;
    VehicleManager *device_vehicle_manager;

    cudaMalloc((void **)&device_vehicle_manager, sizeof(VehicleManager));
    cudaMemcpy(device_vehicle_manager, &host_vehicle_manager,
               sizeof(VehicleManager), cudaMemcpyHostToDevice);

    bool *err;
    cudaMalloc((void **)&err, sizeof(bool));

    check<<<1,1>>>(host_vehicle_manager.size(), device_vehicle_manager, err);

    bool host_err;
    cudaMemcpy(&host_err, err, sizeof(bool), cudaMemcpyDeviceToHost);

    printf("err is %d\n", host_err);

    return 0;
}

/* こうしたいっていうCUDAプログラム
 * thrustの使い方は適当 */
void expect(void)
{
    VehicleManager host_vm;
    VehicleManager *device_vms;

    size_t array_size   = 20;
    size_t device_bytes = array_size * sizeof(VehicleManager);
    cudaMalloc((void **)&device_vms, device_bytes);

    for (int i=0; i < array_size; i++)
        cudaMemcpy(&device_vms[i], &host_vm, sizeof(VehicleManager),
                   cudaMemcpyHostToDevice);

    thrust::device_vector<int> device_costs(array_size);

    randomSimulation<<<array_size, vrp->vertnum>>>(device_vrp, device_vms, device_costs);
    thrust::reduce(device_costs.begin(), device_costs.end(), (int) 0, thrust::min_element<int>());

    thrust::host_vector<int> cost;

    cost = device_costs[0];

    printf("cost %d\n", cost);
}
