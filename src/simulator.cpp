#include <stdlib.h>

extern "C"
{
#include "vrp_types.h"
}

#include "simulator.h"

class Candidates
{
public:
    Candidates(void) : candidate_(), candidate_size_(0) {};
    void collect(const BaseVrp& vrp, VehicleManager& vm);
    int  elect(void);

private:
    int candidate_[200];
    int candidate_size_;
};

void Candidates::collect(const BaseVrp& vrp, VehicleManager& vm)
{
    const int vertnum = vrp.customer_size() + 1;
    for (int i=1; i < vertnum; i++)
        if (!vm.isVisit(i) && vm.canVisitCustomer(vrp, i))
            candidate_[candidate_size_++] = i;
}


int Candidates::elect(void)
{
    if (candidate_size_ == 0)
        return VehicleManager::kChange;
    else
        return candidate_[rand() % candidate_size_];
}

int Simulator::sequentialRandomSimulation(const BaseVrp& vrp, VehicleManager& vm)
{
    while (!vm.isVisitAll(vrp))
    {
        Candidates candidates;

        candidates.collect(vrp, vm);
        int next_move = candidates.elect();

        if (next_move == VehicleManager::kChange &&
            !vm.nextVehicleRemain(vrp))
            return kInfinity;

        vm.move(vrp, next_move);
    }

    return vm.computeTotalCost(vrp);
}

int Simulator::sequentialRandomSimulation(const BaseVrp& vrp, VehicleManager& vm,
                                          int loopCount)
{
    int minCost = kInfinity;
    for (int i=0; i < loopCount; i++)
    {
        VehicleManager vm_copy = vm;
        int cost = sequentialRandomSimulation(vrp, vm_copy);
        if (cost < minCost)
            minCost = cost;
    }
    return minCost;
}

/*
unsigned int Simulator::sequentialRandomSimulation(const BaseVrp& vrp, const Solution solution,
                                                   int count)
{
    unsigned int min_cost = kInfinity;
    for (int i=0; i < count; i++)
    {
        Solution solution_copy = solution;
        int cost = sequentialRandomSimulation(vrp, solution_copy);
        if (cost < min_cost)
            min_cost = cost;
    }
    return min_cost;
}
*/
