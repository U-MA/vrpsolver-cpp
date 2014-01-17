#include <stdio.h>

#include "../include/device_vrp.h"

/* device上のvrpデータを作成して解放する */
void createAndDelete(void)
{
    vrp_problem *device_vrp = createVrpOnDevice();
    deleteVrpOnDevice(device_vrp);
}

void copyCorrectly(void)
{
    vrp_problem *host_vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    host_vrp->vertnum = 100;
    host_vrp->demand  = (int *)calloc(10, sizeof(int));
    host_vrp->demand[0] = 7;

    vrp_problem *device_vrp = createVrpOnDevice();
    copyVrpHostToDevice(device_vrp, host_vrp);

    int vertnum = -1;
    cudaMemcpy(&vertnum, &device_vrp->vertnum, sizeof(int),
               cudaMemcpyDeviceToHost);

    int *device_demand = NULL;
    cudaMemcpy(&device_demand, &device_vrp->demand, sizeof(int *),
               cudaMemcpyDeviceToHost);

    int demand0 = -1;
    cudaMemcpy(&demand0, &device_demand[0], sizeof(int),
               cudaMemcpyDeviceToHost);

    if ((vertnum != host_vrp->vertnum) || (demand0 != host_vrp->demand[0]))
    {
        printf("copyCorrectly is fail\n");
    }
    else
    {
        printf("copyCorrectly is success\n");
    }

    deleteVrpOnDevice(device_vrp);
}

int main(int argc, char **argv)
{
    createAndDelete();
    copyCorrectly();

    return 0;
}
