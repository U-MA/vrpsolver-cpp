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
