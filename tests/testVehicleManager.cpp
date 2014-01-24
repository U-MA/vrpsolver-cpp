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
}

TEST(VehicleManager, canMoveFail)
{
    CHECK_TRUE(vm.canMove(vrp));
}

TEST(VehicleManager, canMoveSuccess)
{
    vm.move(vrp, VehicleManager::kChange);
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    CHECK_FALSE(vm.canMove(vrp));
}
