#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "VrpProblems.h"

#include "simulator.h"
#include "vehicle_manager.h"

TEST_GROUP(Simulation)
{
    vrp_problem    *vrp;
    VehicleManager vm;
    Simulator simulator;
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
    LONGS_EQUAL(206, simulator.sequentialRandomSimulation(vrp, vm));
}

TEST(Simulation, sequentialRandomSimulationWithLoop)
{
    LONGS_EQUAL(171, simulator.sequentialRandomSimulation(vrp, vm, 1000));
}

TEST(Simulation, sequentialRandomSimulationIsMiss)
{
    vm.move(vrp, VehicleManager::kChange);
    LONGS_EQUAL(1e6, simulator.sequentialRandomSimulation(vrp, vm, 20));
}
