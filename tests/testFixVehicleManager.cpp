#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_macros.h"
#include "vrp_types.h"
}

#include "VehicleManager.h"

TEST_GROUP(FixVehicleManager)
{
    vrp_problem *vrp;
    VehicleManager vm;
    Vehicle v;

    void setup()
    {
        vrp            = (vrp_problem *)malloc(sizeof(vrp_problem));
        vrp->dist.cost = (int *)calloc(100, sizeof(int));
        vrp->demand    = (int *)calloc(100, sizeof(int));
    }

    void teardown()
    {
        free(vrp->demand);
        free(vrp->dist.cost);
        free(vrp);
    }

    void Vrp_SetCost(int first, int second, int value)
    {
        vrp->dist.cost[INDEX(first, second)] = value;
    }

    /* Applying to Monte Carlo Techniques to the Capacitated Vehicle
     * Routing Problem Table 2.1, 2.2より */
    void Vrp_SetProblem(void)
    {
        vrp->vertnum = 6;
        vrp->edgenum = vrp->vertnum * (vrp->vertnum-1) / 2;

        Vrp_SetCost(0, 1, 28);
        Vrp_SetCost(0, 2, 31);
        Vrp_SetCost(0, 3, 20);
        Vrp_SetCost(0, 4, 25);
        Vrp_SetCost(0, 5, 34);
        Vrp_SetCost(1, 2, 21);
        Vrp_SetCost(1, 3, 29);
        Vrp_SetCost(1, 4, 26);
        Vrp_SetCost(1, 5, 20);
        Vrp_SetCost(2, 3, 38);
        Vrp_SetCost(2, 4, 20);
        Vrp_SetCost(2, 5, 32);
        Vrp_SetCost(3, 4, 30);
        Vrp_SetCost(3, 5, 27);
        Vrp_SetCost(4, 5, 25);

        vrp->capacity  = 100;
        vrp->demand[1] = 37;
        vrp->demand[2] = 35;
        vrp->demand[3] = 30;
        vrp->demand[4] = 25;
        vrp->demand[5] = 32;
    }
};

TEST(FixVehicleManager, empty)
{
    CHECK_TRUE(vm.empty());
}

TEST(FixVehicleManager, addVehicle)
{
    vm.add(v);
    CHECK_FALSE(vm.empty());
}

TEST(FixVehicleManager, getSize)
{
    LONGS_EQUAL(0, vm.getSize());
    vm.add(v);
    LONGS_EQUAL(1, vm.getSize());
    vm.add(v);
    LONGS_EQUAL(2, vm.getSize());
}

TEST(FixVehicleManager, getVehicle)
{
    Vrp_SetProblem();

    v.visit(vrp, 1);
    CHECK_TRUE(v.isVisitOne(1));
    vm.add(v);
    Vehicle v2 = vm.getVehicle();
    CHECK_TRUE(v2.isVisitOne(1));
}
