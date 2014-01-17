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

    vrp_problem *device_vrp = createVrpOnDevice();
    copyVrpHostToDevice(device_vrp, host_vrp);

    int vertnum = -1;
    cudaMemcpy(&vertnum, &device_vrp->vertnum, sizeof(int),
               cudaMemcpyDeviceToHost);

    if (vertnum != host_vrp->vertnum)
    {
        printf("copyCorrectly is fail\n");
    }

    deleteVrpOnDevice(device_vrp);
}

int main(int argc, char **argv)
{
    createAndDelete();
    copyCorrectly();

    return 0;
}
