#ifndef VRPSOLVER_CPP_DEVICE_VEHICLE_MANAGER_H
#define VRPSOLVER_CPP_DEVICE_VEHICLE_MANAGER_H


/* usage expample
 *
 *
 * int main(int argc, char **argv)
 * {
 *      VehicleManager host_vm;
 *      VehicleManager *device_vm;
 *
 *      cudaMalloc((void **)&device_vm, sizeof(VehicleManager));
 *      cudaMemcpy(device_vm, &host_vm, sizeof(VehicleManager),
 *                 cudaMemcpyHostToDevice);
 *
 *      kernelUsingVehicleManager<<<grids,blocks>>>(device_vm);
 *
 *      // tear down
 *
 *      return 0;
 * }
 *
 *
 * __global__
 * int randomSimulationKernel(const vrp_problem *device_vrp,
 *                            VehicleManger     *device_vms, int count)
 * {
 *      int vms_idx = blockIdx.x; // device_vmsのindex
 *
 *      __shared__ int *candidates;
 *      __shared__ int candidate_size;
 *
 *      while (!device_vms[vms_idx].isVisitAll(device_vrp))
 *      {
 *          if (threadIdx.x == 0)
 *              candidate_size = 0;
 *
 *          __syncthreads();
 *
 *          // 候補者の選定
 *          if (threadIdx.x < device_vrp->vertnum)
 *          {
 *              if (!device_vms[vms_idx].isVisit(threadIdx.x) &&
 *                  device_vms[vms_idx].canVisit(threadIdx.x))
 *              {
 *                  int old = atomicAdd(candidate_size, 1);
 *                  candidates[old] = threadIdx.x;
 *              }
 *          }
 *
 *          __syncthreads();
 *
 *          // 候補者から一人選ぶ
 *          if (threadIdx.x == 0)
 *          {
 *              if (candidate_size == 0)
 *              {
 *                  device_vms[vms_idx].move(device_vrp, VehicleManager::kChange);
 *              }
 *              else
 *              {
 *                  int selected_idx = rand() % candidate_size;
 *                  device_vms[vms_idx].move(device_vrp, candidates[selected_idx]);
 *              }
 *          }
 *
 *          __syncthreads();
 *      }
 *
 *      int cost = 1e6;
 *      if (vm.isVisitAll(device_vrp))
 *          cost = device_vms[vms_idx].computeTotalCost(device_vrp);
 *
 *      return cost;
 *  }
 */

#endif /* VRPSOLVER_CPP_DEVICE_VEHICLE_MANAGER_H */
