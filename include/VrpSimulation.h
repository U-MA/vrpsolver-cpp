#ifndef VRPSOLVER_CPP_SIMULATION_H
#define VRPSOLVER_CPP_SIMULATION_H

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"


#define INF 1e6

namespace VrpSimulation
{
    int sequentialCws(VehicleManager& vm);
    int sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm);
}

#endif /* VRPSOLVER_CPP_SIMULATION_H */
