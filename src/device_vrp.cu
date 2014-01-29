#include "device_vrp.h"

DeviceVrp::DeviceVrp(void)
{
    cudaMalloc((void **)&vrp_, sizeof(vrp_problem));
}

DeviceVrp::DeviceVrp(const HostVrp& host_vrp)
{
    cudaMalloc((void **)&vrp_, sizeof(vrp_problem));

    int vertnum = host_vrp.customer_size()+1;
    cudaMemcpy(&vrp_->vertnum, &vertnum, sizeof(int),
               cudaMemcpyHostToDevice);

    int vehicle_size = host_vrp.vehicle_size();
    cudaMemcpy(&vrp_->numroutes, &vehicle_size, sizeof(int),
               cudaMemcpyHostToDevice);

    int capacity = host_vrp.capacity();
    cudaMemcpy(&vrp_->capacity, &capacity, sizeof(int),
               cudaMemcpyHostToDevice);

    int *host_cost = new int[host_vrp->edge_size()];
    for (int i=0; i < vertnum; i++)
        for (int j=0; j < i; j++)
            host_cost[INDEX(i, j)] = host_vrp->cost(i, j);

    int *device_cost = NULL;
    size_t device_cost_bytes = host_vrp->edge_size() * sizeof(int);
    cudaMalloc((void **)&device_cost, device_cost_bytes);
    cudaMemcpy(device_cost, host_cost, device_cost_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&vrp_->dist.cost, &device_cost, sizeof(int *),
               cudaMemcpyHostToDevice);

    delete[] host_cost;

    int *host_demand = new int[vertnum];
    for (int i=0; i < vertnum; i++)
        host_demand[i] = host_vrp->demand(i);

    int *device_demand = NULL;
    size_t device_demand_bytes = vertnum * sizeof(int);
    cudaMalloc((void **)&device_demand, device_demand_bytes);
    cudaMemcpy(device_demand, host_demand, device_demand_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&vrp_->demand, &device_demand, sizeof(int *),
               cudaMemcpyHostToDevice);

    delete[] host_demand;
}

DeviceVrp::~DeviceVrp(void)
{
    /* TODO */
}

const DeviceVrp& operator=(const HostVrp& host_vrp)
{
    /* TODO */
}
