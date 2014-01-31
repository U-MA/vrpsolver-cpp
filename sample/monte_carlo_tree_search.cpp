/*
 * モンテカルロ木探索のサンプル
 *
 * 失敗しないことを仮定している.失敗すると無限ループ.
 */

#include <iostream>

#include "host_vrp.h"
#include "node.h"
#include "vehicle_manager.h"

int main(int argc, char **argv)
{
    HostVrp host_vrp("../Vrp-All/E/E-n13-k4.vrp");
    VehicleManager vm;

    while (!vm.isVisitAll(host_vrp))
    {
        Node mct;

        for (int i=0; i < 1000; i++)
            mct.build(host_vrp, vm, 100);

        int next_move = mct.selectNextMove();
        if (next_move == VehicleManager::kChange)
            vm.changeVehicle();
        else
            vm.move(host_vrp, next_move);
    }

    vm.print();
    std::cout << "[COST] " << vm.computeTotalCost(host_vrp) << std::endl;

    return 0;
}
