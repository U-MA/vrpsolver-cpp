#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_macros.h"
#include "vrp_types.h"
}

#include "VrpProblems.h"

#include "vehicle_manager.h"

TEST_GROUP(VehicleManager)
{
    VehicleManager vm;
    vrp_problem *vrp;
    Vehicle v;

    void setup()
    {
        vrp = VrpProblem::AMCT2CVRP();
    }

    void teardown()
    {
        VrpProblem::teardown(vrp);
    }
};


TEST(VehicleManager, computeTotalCostAtFirst)
{
    LONGS_EQUAL(0, vm.computeTotalCost(NULL));
}

TEST(VehicleManager, computeTotalCostAfterOneVehicleVisit)
{
    vm.move(vrp, 1);
    LONGS_EQUAL(56, vm.computeTotalCost(vrp));
}

TEST(VehicleManager, computeTotalCostAfterSomeVehicleVisit)
{
    vm.move(vrp, 1);
    vm.move(vrp, VehicleManager::kChange);
    vm.move(vrp, 2);
    LONGS_EQUAL(118, vm.computeTotalCost(vrp));
}

TEST(VehicleManager, notVisitAllCustomer)
{
    CHECK_FALSE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, notVisitAllWhenOneCustomerVisited)
{
    vm.move(vrp, 1);
    CHECK_FALSE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, isVisitAllCustomer)
{
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, VehicleManager::kChange);
    vm.move(vrp, 3);
    vm.move(vrp, 4);
    CHECK_FALSE(vm.isVisitAll(vrp));
    vm.move(vrp, 5);
    CHECK_TRUE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, isVisit)
{
    vm.move(vrp, 1);
    CHECK_TRUE(vm.isVisit(1));

    vm.move(vrp, 5);
    CHECK_TRUE(vm.isVisit(5));
}

TEST(VehicleManager, isNotVisit)
{
    CHECK_FALSE(vm.isVisit(1));
}

TEST(VehicleManager, sizeWhenInit)
{
    LONGS_EQUAL(1, vm.vehicle_size());
}

TEST(VehicleManager, sizeIncrement)
{
    vm.move(vrp, VehicleManager::kChange);
    LONGS_EQUAL(2, vm.vehicle_size());

    CHECK_FALSE(vm.move(vrp, VehicleManager::kChange));
    LONGS_EQUAL(2, vm.vehicle_size());
}

TEST(VehicleManager, moveTrue)
{
    CHECK_TRUE(vm.move(vrp, 1));
}

TEST(VehicleManager, moveFailWhenSameCustomerVisit)
{
    vm.move(vrp, 1);
    CHECK_FALSE(vm.move(vrp, 1));
}

TEST(VehicleManager, moveFailWhenThereIsNoVehicle)
{
    vrp->numroutes = 3;

    CHECK_TRUE(vm.move(vrp, VehicleManager::kChange));
    CHECK_TRUE(vm.move(vrp, VehicleManager::kChange));
    CHECK_FALSE(vm.move(vrp, VehicleManager::kChange));
}

TEST(VehicleManager, moveFailWhenVehicleVisitOverCapacity)
{
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    CHECK_FALSE(vm.move(vrp, 3));
}

TEST(VehicleManager, moveFailWhenSecondVehicleVisitOverCapacity)
{
    vm.move(vrp, 1);
    vm.move(vrp, VehicleManager::kChange);

    CHECK_TRUE(vm.move(vrp, 2));
    CHECK_TRUE(vm.move(vrp, 3));
    CHECK_TRUE(vm.move(vrp, 4));
    CHECK_FALSE(vm.move(vrp, 5));
}

TEST(VehicleManager, canVisit)
{
    CHECK_TRUE(vm.checkCapacityConstraint(vrp, 1));
}

TEST(VehicleManager, CanNotVisit)
{
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    CHECK_FALSE(vm.checkCapacityConstraint(vrp, 3));
}

TEST(VehicleManager, isFinishFail)
{
    CHECK_FALSE(vm.isFinish(vrp));
}

TEST(VehicleManager, isFinishSuccess)
{
    vm.move(vrp, VehicleManager::kChange);
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    CHECK_TRUE(vm.isFinish(vrp));
}
