#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node.h"
#include "solver.h"
#include "vehicle_manager.h"
#include "wrapper_vrp.h"

void Solver::setProblem(const char *filename)
{
    vrp_.Create(filename);
    printf("file name       : %s\n", filename);
}

void Solver::setSeed(long seed)
{
    this->seed_ = seed;
}

void Solver::setMctsIterationCount(int count)
{
    this->count_ = count;
}

void Solver::setSimulationCount(int count)
{
    this->simulation_count_ = count;
}

void Solver::printRunParameter(void)
{
    printf("seed            : %ld\n"  , seed_);
    printf("search count    : %d\n"   , count_);
    printf("simulation count: %d\n\n" , simulation_count_);
}


/* モンテカルロ木探索の実行 */
void Solver::run(void)
{
    printRunParameter();

    VehicleManager vm;
    while (!vm.isVisitAll(vrp_))
    {
        Node mct;

        for (int i=0; i < count_; i++)
            mct.build(vrp_, vm, simulation_count_);

        int next_move = mct.selectNextMove();
        if (next_move == VehicleManager::kChange)
            vm.changeVehicle();
        else
            vm.move(vrp_, next_move);
    }

    vm.print();
    printf("[COST] %6d\n", vm.computeTotalCost(vrp_));
}
