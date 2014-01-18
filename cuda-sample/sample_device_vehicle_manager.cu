#include "VehicleManager.h"

/* DRAM上にVehicleManagerを確保するためのテスト */
int main(int argc, char **argv)
{
    VehicleManager host_vehicle_manager;
    VehicleManager *device_vehicle_manager;

    cudaMalloc((void *)&device_vehicle_manager, sizeof(VehicleManager));
    cudaMemcpy(device_vehicle_manager, &host_vehicle_manager,
               sizeof(VehicleManager), cudaMemcpyHostToDevice);

    return 0;
}
