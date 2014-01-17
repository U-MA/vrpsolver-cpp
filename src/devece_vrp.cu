extern "C"
{
#include "vrp_types.h"
}

#include "device_vrp.h"

vrp_problem *createVrpOnDevice(void)
{
    vrp_problem *device_vrp = NULL;
    cudaMalloc((void **)&device_vrp, sizeof(vrp_problem));
    return device_vrp;
}

static void transferHostToDevice(int **device_member, int *host_member, size_t size_bytes)
{
    int *device_ptr = NULL;
    cudaMalloc((void **)&device_ptr, size_bytes);
    cudaMemcpy(device_ptr, host_member, size_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_member, &device_ptr, sizeof(int *),
               cudaMemcpyHostToDevice);
}

void transferVrpHostToDevice(vrp_problem *device_vrp, const vrp_problem *host_vrp)
{
    cudaMemcpy(device_vrp, host_vrp, sizeof(vrp_problem),
               cudaMemcpyHostToDevice);

    transferHostToDevice(&device_vrp->dist.cost, host_vrp->dist.cost,
                         host_vrp->edgenum * sizeof(int));
    transferHostToDevice(&device_vrp->demand,    host_vrp->demand,
                         host_vrp->vertnum * sizeof(int));
}
