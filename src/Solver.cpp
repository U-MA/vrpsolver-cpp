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
    vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp_io(vrp, filename);
    printf("file name       : %s\n", filename);

    vrp->numroutes = extractVehicleSizeAndToInt(filename);
}

void Solver::setSeed(long seed)
{
    this->seed = seed;
}

void Solver::setMctsIterationCount(int count)
{
    this->count = count;
}

void Solver::setSimulationCount(int count)
{
    this->simulationCount = count;
}

void Solver::cookMember(void)
{
    if (seed == 0)
        seed = 2013;

    if (count == 0)
        count = 1000;

    if (simulationCount == 0)
        simulationCount = 1;

    printf("seed            : %ld\n"  , seed);
    printf("search count    : %d\n"   , count);
    printf("simulation count: %d\n\n" , simulationCount);
}

void Solver::run(void)
{
    cookMember();

    VehicleManager vm;
    while (!vm.isVisitAll(vrp))
    {
        Node mct;

        for (int i=0; i < count; i++)
            mct.search(vrp, vm, simulationCount);

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

void Solver::freeProblem(void)
{
    if (vrp->demand != 0)      free(vrp->demand);
    if (vrp->posx != 0)        free(vrp->posx);
    if (vrp->posy != 0)        free(vrp->posy);
    if (vrp->dist.cost != 0)   free(vrp->dist.cost);
    if (vrp->dist.coordx != 0) free(vrp->dist.coordx);
    if (vrp->dist.coordy != 0) free(vrp->dist.coordy);
    
    free(vrp);
}
