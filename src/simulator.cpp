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
    void correct(const vrp_problem *vrp, VehicleManager &vm);
    int  select(void);

private:
    int candidate_[200];
    int candidate_size_;
};

void Candidates::correct(const vrp_problem *vrp, VehicleManager &vm)
{
    for (int i=1; i < vrp->vertnum; i++)
    {
        if (vm.canVisit(vrp, i))
            candidate_[candidate_size_++] = i;
    }
}

int Candidates::select(void)
{
    if (candidate_size_ == 0)
        return VehicleManager::kChange;
    else
        return candidate_[rand() % candidate_size_];
}


int Simulator::sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm)
{
    while (!vm.isVisitAll(vrp))
    {
        Candidates candidates;

        candidates.correct(vrp, vm);
        int next_move = candidates.select();

        if (!vm.move(vrp, next_move))
            return kInfinity;
    }

    return vm.computeTotalCost(vrp);
}

int Simulator::sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm,
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
