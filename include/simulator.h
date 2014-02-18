#ifndef VRPSOLVER_CPP_SIMULATION_H
#define VRPSOLVER_CPP_SIMULATION_H

#include "base_vrp.h"
#include "vehicle_manager.h"
#include "solution.h"


class Simulator
{
public:
    int sequentialRandomSimulation(const BaseVrp& vrp, VehicleManager& vm);
    int sequentialRandomSimulation(const BaseVrp& vrp, VehicleManager& vm,
                                   int loopCount);
    unsigned int sequentialRandomSimulation(const BaseVrp& vrp, Solution& solution);
    unsigned int sequentialRandomSimulation(const BaseVrp& vrp, const Solution& solution,
                                            unsigned int count);

private:
    static const int kInfinity = 1e6;
};

#endif /* VRPSOLVER_CPP_SIMULATION_H */
