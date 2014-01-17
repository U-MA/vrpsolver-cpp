#include "../include/device_vrp.h"

/* device上のvrpデータを作成して解放する */
int main(int argc, char **argv)
{
    vrp_problem *device_vrp = createVrpOnDevice();

    deleteVrpOnDevice(device_vrp);

    return 0;
}
