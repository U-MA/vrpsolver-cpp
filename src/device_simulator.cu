#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "device_simulator.h"
#include "device_vrp.h"


__global__
void simulation_kernel(DeviceVrp *, VehicleManager *, int *);


/* 引数のVehicleManagerはホスト上に配置されている */
DeviceSimulator::Run(const HostVrp& host_vrp, const VehicleManager& vm, int count)
{
    DeviceVrp device_vrp(host_vrp);

    thrust::device_vector<VehicleManager> device_vms(count);

    /* vmの内容をdevice_vmsにコピー */
    thrust::generate(device_vms.begin(), device_vms.end(), vm);

    thrust::device_vector<int> device_costs(count);

    VehicleManager *device_vms_ptr = thrust::raw_pointer_cast(device_vms);
    int *device_costs_ptr          = thrust::raw_pointer_cast(device_costs);

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
    extern __shared__ int candidates[];
    __shared__ int  candidate_size;
    __shared__ bool isFail;

    philox4x32_key_t key = {{ blockIdx.x, 0xdeadbeaf }};
    philox4x32_ctr_t ctr = {{ 0, 0xf00dcafe, 0xdeadbead, 0xbeaff00d }};

    if (threadIdx.x == 0)
    {
        isFail = false;
        candidate_size = 0;
    }
    __syncthreads();

    const int vertnum = device_vrp->customer_size()+1;
    while(isFail || !device_vms[blockIdx.x].isFinish(device_vrp))
    {
        /* 候補者の選定 */
        if ((0 < threadIdx.x) && (threadIdx.x < vertnum) &&
            !device_vms[blockIdx.x].isVisit(threadIdx.x) &&
            device_canVisitCustomer(device_vms[blockIdx.x], threadIdx.x))
        {
            int old = atomicAdd(&candidates_size, 1);
            candidates[old] = threadIdx.x;
        }
        __syncthreads();

        /* 候補者を一人選ぶ */
        if (threadIdx.x == 0)
        {
            if (candidate_size == 0)
            {
                if (!device_vms[blockIdx.x].nextVehicleRemain(device_vrp))
                {
                    isFail = !isFail;
                }
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
                int selected = candidate[u.c[0] % candidate_size];
                device_vms[blockIdx.x].move(device_vrp, selected);
            }
            candidate_size = 0;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (device_vms[blockIdx.x].isVisitAll(device_vrp))
            device_costs[blockIdx.x] = device_vms[blockIdx.x].computeTotalCost(device_vrp);
        else
            device_costs[blockIdx.x] = 1000000;
    }
}
