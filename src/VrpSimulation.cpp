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

static bool isVisitable(const VehicleManager *vm, const Vehicle *v, const vrp_problem *vrp, int customer)
{
    return (!v->isVisitOne(customer) && !vm->isVisitOne(customer) &&
            v->getQuantity() + vrp->demand[customer] <= vrp->capacity);
}

int VrpSimulation::sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm)
{
    vector<int> candidates;
    Vehicle runVehicle; /* 現在作業している車体 */
    runVehicle.init();

    /* 全ての顧客を訪れるか,使える車体が無くなるまで繰り返す */
    while (!vm.isVisitAll(vrp) && vm.size() < vrp->numroutes)
    {
        /* 次に訪れる顧客の候補を調べる */
        for (int i=1; i < vrp->vertnum; i++)
            if (isVisitable(&vm, &runVehicle, vrp, i))
                candidates.push_back(i);

        if (candidates.empty())
        {
            /* 候補がいなければ次の車体へ移る */
            vm.add(runVehicle);
            runVehicle.init();
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

    int cost = INF;
    if (vm.isVisitAll(vrp))
        cost = vm.computeTotalCost(vrp);

    return cost;
}
