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

void Solver::setProblem(char *filename)
{
    vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
    vrp_io(vrp, filename);
    printf("file name       : %s\n", filename);

    /* numroutesの設定 */
    char *k   = strrchr(filename, 'k');
    char *dot = strrchr(filename, '.');
    char numVehicle[3];
    int n = (dot-k)/sizeof(char);
    strncpy(numVehicle, k+1, n);
    numVehicle[n+1] = '\0';
    vrp->numroutes = atoi(numVehicle);
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
