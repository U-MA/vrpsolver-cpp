extern "C"
{
#include "vrp_types.h"
}

#include "SavingsList.h"
#include "VrpSimulation.h"

//int VrpSimulation::SequentialCws(VehicleManager& vm, const vrp_problem *vrp)
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
