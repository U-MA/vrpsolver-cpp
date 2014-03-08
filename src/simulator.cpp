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

unsigned int Simulator::sequentialRandomSimulation(const BaseVrp& vrp, Solution& solution)
{
    Vehicle *current_vehicle = solution.CurrentVehicle();
    int candidates[200], candidate_size;

    while (!solution.IsFinish())
    {
        candidate_size = 0;

        /* 次に訪問する顧客の候補を求める */
        for (int i=1; i <= vrp.customer_size(); i++)
        {
            /* 訪問可能であれば候補に追加 */
            if (!solution.IsVisit(i) &&
                current_vehicle->Capacity() + (unsigned)vrp.demand(i) <= (unsigned)vrp.capacity())
            {
                candidates[candidate_size++] = i;
            }
        }

        if (candidate_size == 0)
        {
            /* 候補がいなければ次の車両へ */
            solution.ChangeVehicle();
            current_vehicle = solution.CurrentVehicle();
        }
        else
        {
            /* 候補の中から無作為に１人選ぶ */
            int customer = candidates[rand() % candidate_size];
            current_vehicle->Visit(vrp, customer);
        }
    }

    if (solution.IsFeasible())
        return solution.ComputeTotalCost(vrp);
    else
        return kInfinity;
}

unsigned int Simulator::sequentialRandomSimulation(const BaseVrp& vrp, const Solution& solution,
                                                   unsigned int count)
{
    unsigned int min_cost = kInfinity;
    for (unsigned int i=0; i < count; i++)
    {
        Solution solution_copy;
        solution.Copy(solution_copy);
        unsigned int cost = sequentialRandomSimulation(vrp, solution_copy);
        if (cost < min_cost)
            min_cost = cost;
    }
    return min_cost;
}

unsigned int Simulator::random(const BaseVrp& vrp, Solution& solution)
{
    return 10000000;
}
