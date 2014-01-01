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
        vm = VehicleManager(1);

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

TEST(VehicleManager, Init)
{
    LONGS_EQUAL(0, vm.getRunningVehicleNumber());
}

TEST(VehicleManager, ChangeVehicle)
{
    vm = VehicleManager(2);

    vm.changeVehicle();
    LONGS_EQUAL(1, vm.getRunningVehicleNumber());
}

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
    vm.update(vrp, 1);
    LONGS_EQUAL(60, vm.computeTotalCost(vrp));
}

TEST(VehicleManager, computeTotalCostAfterSomeVehicleVisit)
{
    vm = VehicleManager(2);

    vm.update(vrp, 1);
    vm.changeVehicle();
    vm.update(vrp, 2);
    LONGS_EQUAL(100, vm.computeTotalCost(vrp));
}

TEST(VehicleManager, updateOutOfCustomer)
{
    CHECK_FALSE(vm.update(vrp, 0));
    CHECK_FALSE(vm.update(vrp, -1));
    CHECK_FALSE(vm.update(vrp, 10));
}

TEST(VehicleManager, updateInCustomer)
{
    CHECK_TRUE(vm.update(vrp, 1));
    CHECK_TRUE(vm.update(vrp, 9));
}

TEST(VehicleManager, randomSimulation)
{
    vrp->vertnum = 3;
    vm = VehicleManager(2);

    srand(2013);
    LONGS_EQUAL(100, vm.randomSimulation(vrp));
}

TEST(VehicleManager, notVisitAllCustomer)
{
    CHECK_FALSE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, visitAllCustomer)
{
    vrp->vertnum = 3;
    
    vm.update(vrp, 1);
    vm.changeVehicle();
    vm.update(vrp, 2);
    CHECK_TRUE(vm.isVisitAll(vrp));
}

TEST(VehicleManager, isVisit)
{
    vm.update(vrp, 1);
    CHECK_TRUE(vm.isVisitOne(1));
    CHECK_FALSE(vm.isVisitOne(2));
}

TEST(VehicleManager, getEmptyVehicle)
{
    LONGS_EQUAL(0, vm.getEmptyVehicle());
}

TEST(VehicleManager, getEmptyVehicle2)
{
    vm = VehicleManager(2);
    vm.update(vrp, 1);
    LONGS_EQUAL(1, vm.getEmptyVehicle());
}

TEST(VehicleManager, DoNotgetEmptyVehicle)
{
    vm.update(vrp, 1);
    LONGS_EQUAL(-1, vm.getEmptyVehicle());
}
