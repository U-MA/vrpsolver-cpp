#include <vector>


extern "C"
{
#include "vrp_types.h"
}

#include "SavingsList.h"
#include "VrpSimulation.h"

//int VrpSimulation::sequentialCws(VehicleManager& vm, const vrp_problem *vrp)
//{
//    SavingsList savingsList(vrp);

//    while (savingsList.getSize() != 0)
//    {
//        EDGE edge = savingsList.getEdge();

//        if (!vm.isVisit(edge.first) && !vm.isVisit(edge.second))
//        {
            /* どちらの顧客も訪問していない */
//        }
//        else if (vm.isVisit(edge.first) && !vm.isVisit(edge.second))
//        {
            /* firstのみがいずれかの車体のルートに含まれている */
//        }
//        else if (!vm.isVisit(edge.first) && vm.isVisit(edge.second))
//        {
            /* secondのみがいずれかの車体のルートに含まれている */
//        }
//        else
//        {
            /* どちらの顧客も異なる車体のルートに含まれている */
//        }
//    }

//    return vm.computeTotalCost(vrp);
//}

int VrpSimulation::sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm)
{
    vector<int> candidates;
    Vehicle runVehicle; /* 現在作業している車体 */
    runVehicle.init();

    int vehicleCount = 0;
    /* 全ての顧客を訪れるか,使える車体が無くなるまで繰り返す */
    while (!vm.isVisitAll(vrp) && vehicleCount < vrp->numroutes)
    {
        /* 次に訪れる顧客の候補を調べる */
        for (int i=1; i < vrp->vertnum; i++)
        {
            if (!runVehicle.isVisitOne(i) && !vm.isVisitOne(i) &&
                runVehicle.getQuantity() + vrp->demand[i] <= vrp->capacity)
            {
                candidates.push_back(i);
            }
        }

        if (candidates.empty())
        {
            /* 候補がいなければ次の車体へ移る */
            vm.add(runVehicle);
            runVehicle.init();
            vehicleCount++;
        }
        else
        {
            /* 候補の中から一人無作為に選び,選ばれた候補を訪問 */
            int nextCustomer = candidates[rand() % candidates.size()];
            runVehicle.visit(vrp, nextCustomer);
        }

        /* 候補者のリセット */
        candidates.clear();
    }

    int cost = 1000000;
    if (vm.isVisitAll(vrp))
    {
        cost = vm.computeTotalCost(vrp);
    }

    vm.print();

    return cost;
}
