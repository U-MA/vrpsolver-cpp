#ifndef VRPSOLVER_CPP_SIMULATION_H
#define VRPSOLVER_CPP_SIMULATION_H

extern "C"
{
#include "vrp_types.h"
}

#include "base_vrp.h"
#include "vehicle_manager.h"


class Simulator
{
public:
    int sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm);
    int sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm,
                                   int loopCount);
    int sequentialRandomSimulation(const BaseVrp& vrp, VehicleManager& vm);
    int sequentialRandomSimulation(const BaseVrp& vrp, VehicleManager& vm,
                                   int loopCount);

private:
    static const int kInfinity = 1e6;
};

#endif /* VRPSOLVER_CPP_SIMULATION_H */
