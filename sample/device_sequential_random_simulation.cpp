/*
 * CUDAを使ったシミュレーションの期待
 */



#include <iostream>
#include <cstdlib>

#include "host_vrp.h"
#include "device_vrp.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "usage: " << argv[0] << "<simulation_count>" << endl;
        exit(1);
    }

    HostVrp host_vrp("../Vrp-All/E/E-n13-k4.vrp");
    HostVehicleManager host_vm;

    DeviceSimulator device_simulator;
    int simulation_count = atoi(argv[1]);
    int cost = device_simulator.SequentialRandom(host_vrp, device_vrp, 2048);

    cout << "NAME: " << host_vrp.name() << endl;
    cout << "SIMULATION COUNT: " << simulation_count << endl;
    cout << "COST: " << cost << endl;

    return 0;
}
