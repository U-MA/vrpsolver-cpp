#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "Random123/philox.h"

extern "C"
{
#include "vrp_io.h"
#include "vrp_types.h"
}

#include "device_simulation.h"
#include "vehicle_manager.h"
#include "simulator.h"


__global__
void testTransfer(vrp_problem *device_vrp, VehicleManager *device_vms,
		      int *device_costs)
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

        device_vms[0].move(device_vrp, VehicleManager::kChange);
        device_vms[0].print();
    }
}

__device__
bool isValidCustomer(int customer, int customer_size)
{
    return (0 < customer) && (customer <= customer_size);
}

__global__
void randomSimulation(vrp_problem *device_vrp, VehicleManager *device_vms,
                      int *device_costs)
{
    extern __shared__ int candidates[];
    __shared__ int  candidate_size;
    __shared__ bool isFail;

    const int bid      = blockIdx.x;
    const int customer = threadIdx.x;

    philox4x32_key_t key = {{ blockIdx.x, 0xdeadbeef }};
    philox4x32_ctr_t ctr = {{ 0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d }};
    
    if (threadIdx.x == 0)
    {
        isFail = false;
        candidate_size = 0;
    }
    __syncthreads();

    while (isFail || !device_vms[bid].isFinish(device_vrp))
    {
        /* Œó•âŽÒ‚Ì‘I’è */
        if (isValidCustomer(customer, device_vrp->vertnum-1) && !device_vms[bid].isVisit(customer) &&
            device_vms[bid].canVisit(device_vrp, customer))
        {
            int old = atomicAdd(&candidate_size, 1);
            candidates[old] = customer;
        }
        __syncthreads();

        /* Œó•âŽÒ‚ðˆêl‘I‚Ô */
        if (threadIdx.x == 0)
        {
            if (candidate_size == 0)
            {
                if (!device_vms[bid].move(device_vrp, VehicleManager::kChange))
                    isFail = true;
            }
            else
            {
                union
                {
                    philox4x32_ctr_t c;
                    int4 i;
                } u;
                ctr.v[0]++;

                u.c = philox4x32(ctr, key);
                int selected = candidates[u.c[0] % candidate_size];
                device_vms[bid].move(device_vrp, selected);
            }
            candidate_size = 0;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (device_vms[bid].isVisitAll(device_vrp))
            device_costs[bid] = device_vms[bid].computeTotalCost(device_vrp);
        else
            device_costs[bid] = 100000;
    }
}

int main(int argc, char **argv)
{
    char infile[200];
    strcpy(infile, "Vrp-All/E/E-n101-k14.vrp");
    
    vrp_problem *host_vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp_io(host_vrp, infile);
    host_vrp->numroutes = 14;

    VehicleManager vm;
    int min_cpu = 100000;
    clock_t start_cpu, end_cpu;

    Simulator s;
    start_cpu = clock();
    min_cpu = s.sequentialRandomSimulation(host_vrp, vm, 2048);
    end_cpu   = clock();
    std::cout << "min cost(CPU) " << min_cpu << std::endl;
    std::cout << "time(CPU)     " << (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC << "sec" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
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

    int parallel_size = 2048;
    cudaMalloc((void **)&device_vms, parallel_size * sizeof(VehicleManager));
    for (int i=0; i < parallel_size; i++)
        cudaMemcpy(&device_vms[i], &host_vm, sizeof(VehicleManager),
                   cudaMemcpyHostToDevice);

    thrust::device_vector<int> device_vector(parallel_size);
    int *device_costs = thrust::raw_pointer_cast(device_vector.data());

    size_t shared_bytes = host_vrp->vertnum * sizeof(int);
    randomSimulation<<<parallel_size,host_vrp->vertnum,shared_bytes>>>(device_vrp, device_vms, device_costs);
    cudaDeviceSynchronize();

    int min = thrust::reduce(device_vector.begin(), device_vector.end(), (int) 1e6,
                             thrust::minimum<int>());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    std::cout << "min cost " << min << std::endl;
    float elapsed_time_ms = .0f;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "time: " << elapsed_time_ms << "ms ( " << elapsed_time_ms / 1000 << "sec )" << std::endl;

    /* todo cudaFree */

    return 0;
}
