#include <string.h>

extern "C"
{
#include "vrp_io.h"
#include "vrp_types.h"
}

#include "Node.h"
#include "Solver.h"
#include "VehicleManager.h"

vrp_problem *vrp;
long gSeed;
int  gCount;
int  gSimulationCount;


void Solver::setProblem(char *filename)
{
    vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
    vrp_io(vrp, filename);

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
    gSeed = seed;
}

void Solver::setIteration(int count)
{
    gCount = count;
}

void Solver::setSimulationLoop(int count)
{
    gSimulationCount = count;
}

void Solver::run(void)
{
    VehicleManager vm;

    while (!vm.isVisitAll(vrp))
    {
        Node mct;

        for (int i=0; i < gCount; i++)
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
