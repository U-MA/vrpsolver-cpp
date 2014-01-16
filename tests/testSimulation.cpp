#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "VrpProblems.h"

#include "SavingsList.h"
#include "VrpSimulation.h"
#include "VehicleManager.h"

TEST_GROUP(Simulation)
{
    vrp_problem    *vrp;
    VehicleManager vm;
    void setup()
    {
        srand(2013);
        vrp = VrpProblem::AMCT2CVRP();
    }

    void teardown()
    {
        VrpProblem::teardown(vrp);
    }
};

/* 乱数を使用するため,適切なテスト方法がわからない.
 * なので関数の見た目等を考えるための記述を行う */

TEST(Simulation, sequenatialRandomSimulation)
{
    LONGS_EQUAL(206, VrpSimulation::sequentialRandomSimulation(vrp, vm));
}

TEST(Simulation, sequentialRandomSimulationWithLoop)
{
    LONGS_EQUAL(171, VrpSimulation::sequentialRandomSimulation(vrp, vm, 1000));
}

TEST(Simulation, sequentialRandomSimulationIsMiss)
{
    vm.move(vrp, VehicleManager::kChange);
    LONGS_EQUAL(VrpSimulation::kInfinity, VrpSimulation::sequentialRandomSimulation(vrp, vm, 20));
}
