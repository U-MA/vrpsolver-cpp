#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node.h"
#include "solver.h"
#include "vehicle_manager.h"
#include "wrapper_vrp.h"

Solver::~Solver(void)
{
    destroyVrp(vrp_);
}

void Solver::setProblem(char *filename)
{
    vrp_ = createVrpFromFilePath(filename);
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
            mct.search(vrp_, vm, simulation_count_);

        int move = mct.selectNextMove();

        /* 用意している車体数を超えるとbreak */
        if (!vm.move(vrp_, move))
            break;
    }

    int cost = 1e6;
    if (vm.isVisitAll(vrp_))
        cost = vm.computeTotalCost(vrp_);

    vm.print();
    printf("[COST] %6d\n", cost);
}
