#ifndef VRPSOLVER_CPP_SIMULATION_H
#define VRPSOLVER_CPP_SIMULATION_H

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"


namespace VrpSimulation
{
    static const int kInfinity = 1e6;
    int sequentialCws(const vrp_problem *vrp, VehicleManager& vm);
    int sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm);
    int sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm, int loopCount);
}

#endif /* VRPSOLVER_CPP_SIMULATION_H */
