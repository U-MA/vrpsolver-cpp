/*
 * ちゃんと使えるかを意図したファイル
 *
 * 将来の使い方とは異なることがあり得る.というか変える.
 */

#include <iostream>
#include <cstdlib>

#include "host_vrp.h"
#include "simulator.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "usage: " << argv[0] << "<simulation_count>" << endl;
        exit(1);
    }

    HostVrp host_vrp("../Vrp-All/E/E-n13-k4.vrp");
    VehicleManager vm;

    Simulator simulator;
    int simulation_count = atoi(argv[1]);

    cout << "NAME: " << host_vrp.name() << endl;
    cout << "SIMULATION COUNT: " << simulation_count << endl;
    cout << "COST: " <<
            simulator.sequentialRandomSimulation(host_vrp, vm,
                                                 simulation_count) << endl;

    return 0;
}
