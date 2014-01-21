#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

extern "C"
{
#include "vrp_io.h"
#include "vrp_types.h"
}

#include "device_simulation.h"
#include "vehicle_manager.h"


__global__
void testTransfer(vrp_problem *device_vrp, VehicleManager *device_vms,
		      thrust::device_vector<int> device_costs)
{
    if (threadIdx.x + blockIdx.x == 0)
    {
        printf("CHECK device_vrp MEMBER\n");
	    printf("\tdevice_vrp->vertnum      %d\n",
               device_vrp->vertnum);
        printf("\tdevice_vrp->demand[0]    %d\n"
               "\tdevice_vrp->demand[3]    %d\n",
               device_vrp->demand[0], device_vrp->demand[3]);
        printf("\tdevice_vrp->dist.cost[0] %d\n",
               device_vrp->dist.cost[0]);

        printf("CHECK device_vms\n");
        printf("\tdevice_vms[0].size()     %d\n",
               device_vms[0].size());
        printf("\tdevice_vms[99].siz()     %d\n",
               device_vms[99].size());
        printf("\tdevice_vms[0]'s cost     %d\n",
               device_vms[0].computeTotalCost(device_vrp));
        printf("\tdevice_vms[99]'s cost    %d\n",
               device_vms[99].computeTotalCost(device_vrp));

        printf("\tdevice_vms[0] move customer 1\n");
        device_vms[0].move(device_vrp, 1);

        printf("\tdevice_vms[0]'s cost     %d\n",
               device_vms[0].computeTotalCost(device_vrp));
        printf("\tdevice_vms[99]'s cost    %d\n",
               device_vms[99].computeTotalCost(device_vrp));
    }
}

__global__
void randomSimulation(vrp_problem *vrp, VehicleManager *device_vms,
                      thrust::device_vector<int> device_costs)
{
    __shared__ int *candidates;
    __shared__ int candidate_size;

    int bid      = blockIdx.x;
    int customer = threadIdx.x;

    while (device_vms[bid].isFinish(device_vrp))
    {
    }
}



int main(int argc, char **argv)
{
    char infile[200];
    strcpy(infile, "Vrp-All/E/E-n13-k4.vrp");
    
    vrp_problem *host_vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp_io(host_vrp, infile);

    vrp_problem *device_vrp = NULL;
    cudaMalloc((void **)&device_vrp, sizeof(vrp_problem));

    cudaMemcpy(device_vrp, host_vrp, sizeof(vrp_problem),
               cudaMemcpyHostToDevice);

    int    *device_cost = NULL;
    size_t device_cost_bytes = host_vrp->edgenum * sizeof(int);
    cudaMalloc((void **)&device_cost, device_cost_bytes);
    cudaMemcpy(device_cost, host_vrp->dist.cost, device_cost_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&device_vrp->dist.cost, &device_cost, sizeof(int *),
               cudaMemcpyHostToDevice);

    int *device_demand = NULL;
    size_t device_demand_bytes = host_vrp->vertnum * sizeof(int);
    cudaMalloc((void **)&device_demand, device_demand_bytes);
    cudaMemcpy(device_demand, host_vrp->demand, device_demand_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&device_vrp->demand, &device_demand, sizeof(int *),
               cudaMemcpyHostToDevice);


    VehicleManager host_vm;
    VehicleManager *device_vms;

    cudaMalloc((void **)&device_vms, 100 * sizeof(VehicleManager));
    for (int i=0; i < 100; i++)
        cudaMemcpy(&device_vms[i], &host_vm, sizeof(VehicleManager),
                   cudaMemcpyHostToDevice);

    thrust::device_vector<int> device_vector(100);

    testTransfer<<<100, host_vrp->vertnum>>>(device_vrp, device_vms, device_vector);

    int min = thrust::reduce(device_vector.begin(), device_vector.end(), (int) 1e6,
                             thrust::minimum<int>());

    std::cout << "min cost " << min << std::endl;

    return 0;
}
