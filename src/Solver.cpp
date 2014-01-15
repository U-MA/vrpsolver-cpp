#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C"
{
#include "vrp_io.h"
#include "vrp_types.h"
}

#include "Node.h"
#include "Solver.h"
#include "VehicleManager.h"

/* filename中のk以下の数字文字列を取り出し、整数値に変換 */
static int extractVehicleSizeAndToInt(char *filename)
{
    char *k   = strrchr(filename, 'k');
    char *dot = strrchr(filename, '.');
    int  n    = (dot-k) / sizeof(char);

    char vehicle_size[3];
    strncpy(vehicle_size, k+1, n);
    vehicle_size[n+1] = '\0';
    return atoi(vehicle_size);
}

void Solver::setProblem(char *filename)
{
    vrp_ = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp_io(vrp_, filename);
    vrp_->numroutes = extractVehicleSizeAndToInt(filename);

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

void Solver::run(void)
{
    printf("seed            : %ld\n"  , seed_);
    printf("search count    : %d\n"   , count_);
    printf("simulation count: %d\n\n" , simulation_count_);

    VehicleManager vm;
    while (!vm.isVisitAll(vrp_))
    {
        Node mct;

        for (int i=0; i < count_; i++)
            mct.search(vrp_, vm, simulation_count_);

        int move = mct.next();

        if (!vm.move(vrp_, move))
            break;
    }

    int cost = 1e6;
    if (vm.isVisitAll(vrp_))
        cost = vm.computeTotalCost(vrp_);

    vm.print();
    printf("[COST] %6d\n", cost);
}

void Solver::freeProblem(void)
{
    if (vrp_->demand      != 0) free(vrp_->demand);
    if (vrp_->posx        != 0) free(vrp_->posx);
    if (vrp_->posy        != 0) free(vrp_->posy);
    if (vrp_->dist.cost   != 0) free(vrp_->dist.cost);
    if (vrp_->dist.coordx != 0) free(vrp_->dist.coordx);
    if (vrp_->dist.coordy != 0) free(vrp_->dist.coordy);
    
    free(vrp_);
}
