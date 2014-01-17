#include "../include/device_vrp.h"

/* device上のvrpデータを作成して解放する */
void createAndDelete(void)
{
    vrp_problem *device_vrp = createVrpOnDevice();
    deleteVrpOnDevice(device_vrp);
}

int main(int argc, char **argv)
{
    createAndDelete();

    return 0;
}
