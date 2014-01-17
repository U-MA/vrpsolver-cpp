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

    deleteVrpOnDevice(device_vrp);
}

int main(int argc, char **argv)
{
    createAndDelete();
    copyCorrectly();

    return 0;
}
