#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

TEST_GROUP(VehicleManager)
{
    VehicleManager vm;
    vrp_problem *vrp;

    void setup()
    {
        vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
        vrp->vertnum = 10;
        vrp->dist.cost = (int *)calloc(10, sizeof(int));
        vrp->demand = (int *)calloc(10, sizeof(int));

        vrp->capacity = 2000;

        vrp->demand[1] = 1900;
        vrp->demand[2] = 1100;

        vrp->dist.cost[0] = 30; /* 0-1 */
        vrp->dist.cost[1] = 20; /* 0-2 */
        vrp->dist.cost[2] = 15; /* 1-2 */
    }

    void teardown()
    {
        free(vrp->demand);
        free(vrp->dist.cost);
        free(vrp);
    }
};


TEST(VehicleManager, ChangeNextLastVehicle)
{
    CHECK_FALSE(vm.changeVehicle());
}

TEST(VehicleManager, computeTotalCostAtFirst)
{
    LONGS_EQUAL(0, vm.computeTotalCost(NULL));
}

TEST(VehicleManager, computeTotalCostAfterOneVehicleVisit)
{
    Vehicle v;
    v.visit(vrp, 1);
    vm.add(v);
    LONGS_EQUAL(60, vm.computeTotalCost(vrp));
}

TEST(VehicleManager, computeTotalCostAfterSomeVehicleVisit)
{
    Vehicle v1, v2;
    v1.visit(vrp, 1);
    v2.visit(vrp, 2);
    vm.add(v1);
    vm.add(v2);
    LONGS_EQUAL(100, vm.computeTotalCost(vrp));
}

TEST(VehicleManager, notVisitAllCustomer)
{
    CHECK_FALSE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, visitAllCustomer)
{
    vrp->vertnum = 3;

    Vehicle v1, v2;
    v1.visit(vrp, 1);
    v2.visit(vrp, 2);
    vm.add(v1);
    vm.add(v2);
    CHECK_TRUE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, isVisit)
{
    Vehicle v;
    v.visit(vrp, 1);
    vm.add(v);
    CHECK_TRUE(vm.isVisitOne(1));
    CHECK_FALSE(vm.isVisitOne(2));
}

TEST(VehicleManager, size)
{
    LONGS_EQUAL(0, vm.size());
}
