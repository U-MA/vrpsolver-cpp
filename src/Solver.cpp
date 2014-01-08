extern "C"
{
#include "vrp_types.h"
}

#include "Node.h"
#include "Solver.h"
#include "VehicleManager.h"

vrp_problem *vrp;

void Solver::setSeed(long seed)
{
    srand(seed);
}

void Solver::run(void)
{
    VehicleManager vm;

    while (!vm.isVisitAll(vrp))
    {
        Node mct;

        for (int i=0; i < 1000; i++)
            mct.search(vrp, vm);

        int move = mct.next();

        if (!vm.move(vrp, move))
            break;
    }

    int cost = 1e6;
    if (vm.isVisitAll(vrp))
        cost = vm.computeTotalCost(vrp);

    vm.print();
    printf("[COST] %6d\n", cost);
}
