#include <stdlib.h>

extern "C"
{
#include "vrp_types.h"
}

#include "SavingsList.h"
#include "VrpSimulation.h"

int VrpSimulation::sequentialCws(const vrp_problem *vrp, VehicleManager& vm)
{
    SavingsList savingsList(vrp);

    while (savingsList.getSize() != 0)
    {
        if (true)
        {
            /* どちらの顧客も訪問していない */
        }
        else if (false)
        {
            /* firstのみがいずれかの車体のルートに含まれている */
        }
        else if (false)
        {
            /* secondのみがいずれかの車体のルートに含まれている */
        }
        else
        {
            /* どちらの顧客も異なる車体のルートに含まれている */
        }
    }

    return vm.computeTotalCost(vrp);
}

static bool isVisitable(const VehicleManager *vm, const Vehicle *v, const vrp_problem *vrp, int customer)
{
    return (!(v->isVisit(customer) || vm->isVisit(customer)) &&
            v->quantity() + vrp->demand[customer] <= vrp->capacity);
}

int VrpSimulation::sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm, Vehicle& v)
{
    Vehicle runVehicle; /* 現在作業している車体 */
    runVehicle.init();

    /* 全ての顧客を訪れるか,使える車体が無くなるまで繰り返す */
    while (!vm.isVisitAll(vrp) && vm.size() < vrp->numroutes)
    {
        int candidates[200], candidatesSize = 0;
        /* 次に訪れる顧客の候補を調べる */
        for (int i=1; i < vrp->vertnum; i++)
            if (isVisitable(&vm, &runVehicle, vrp, i))
            {
                candidates[candidatesSize] = i;
                candidatesSize++;
            }

        if (candidatesSize == 0)
        {
            /* 候補がいなければ次の車体へ移る */
            vm.move(vrp, VehicleManager::CHANGE);
            runVehicle.init();
        }
        else
        {
            /* 候補の中から一人無作為に選び,選ばれた候補を訪問 */
            int nextCustomer = candidates[rand() % candidatesSize];
            runVehicle.visit(vrp, nextCustomer);
        }
    }

    int cost = INF;
    if (vm.isVisitAll(vrp))
        cost = vm.computeTotalCost(vrp);

    return cost;
}

int VrpSimulation::sequentialRandomSimulation(const vrp_problem *vrp, VehicleManager& vm)
{
    while (!vm.isVisitAll(vrp))
    {
        int candidates[200], candidatesSize = 0;

        /* 次に選ばれる顧客の候補を調べる */
        for (int i=1; i < vrp->vertnum; i++)
        {
            if (!vm.isVisit(i) && vm.canVisit(vrp, i))
                candidates[candidatesSize++] = i;
        }

        if (candidatesSize == 0)
        {
            /* 候補がいなければ次の車体へ
             * 但し, moveが失敗するとbreak */
            if (!vm.move(vrp, VehicleManager::CHANGE))
                break;
        }
        else
        {
            /* 候補の中から無作為に一つ選び、その動作を行う */
            int nextCustomer = candidates[rand() % candidatesSize];
            vm.move(vrp, nextCustomer);
        }
    }

    int cost = INF;
    if (vm.isVisitAll(vrp))
        cost = vm.computeTotalCost(vrp);

    return cost;
}
