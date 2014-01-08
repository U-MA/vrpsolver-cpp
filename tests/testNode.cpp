#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "Node.h"
#include "VehicleManager.h"
#include "VrpSimulation.h"


TEST_GROUP(Node)
{
    vrp_problem *vrp;
    Node node;
    void setup()
    {
        vrp            = (vrp_problem *)malloc(sizeof(vrp_problem));
        vrp->dist.cost = (int *)calloc(100, sizeof(int));
        vrp->demand    = (int *)calloc(100, sizeof(int));

        srand(2013);
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

        vrp->numroutes = 2;
        vrp->capacity  = 100;
        vrp->demand[1] = 37;
        vrp->demand[2] = 35;
        vrp->demand[3] = 30;
        vrp->demand[4] = 25;
        vrp->demand[5] = 32;
    }
};

TEST(Node, create)
{
    LONGS_EQUAL(0, node.customer());
    LONGS_EQUAL(0, node.count());
    LONGS_EQUAL(0, node.childSize());
    LONGS_EQUAL(0, node.value());
}

TEST(Node, isLeaf)
{
    CHECK_TRUE(node.isLeaf());

    node.expand(1);
    CHECK_FALSE(node.isLeaf());
}

TEST(Node, createChild)
{
    node.expand(5);
    LONGS_EQUAL(5, node.childSize());
}

TEST(Node, selectChildWhenNodeDontHaveChild)
{
    Node *selected = node.select();
    POINTERS_EQUAL(NULL, selected);
}

TEST(Node, selectChild)
{
    node.expand(1);
    Node *selected = node.select();
    LONGS_EQUAL(0, selected->customer());
}

TEST(Node, selectChildWithMaxUcb)
{
    node.expand(2);
    Node *selected = node.select();
    selected->update(100);
    selected = node.select();
    LONGS_EQUAL(0, selected->customer());
}

TEST(Node, update)
{
    node.expand(1);
    Node *selected = node.select();

    selected->update(100);
    LONGS_EQUAL(1, selected->count());
    LONGS_EQUAL(100, selected->value());

    selected->update(30);
    LONGS_EQUAL(2, selected->count());
    LONGS_EQUAL(130, selected->value());
}

TEST(Node, expandWithVehicleManager)
{
    Vrp_SetProblem();

    VehicleManager vm;

    node.expand(vrp, vm);
    LONGS_EQUAL(6, node.childSize());
}

TEST(Node, expandAfterVehicleVisitOneCustomer)
{
    Vrp_SetProblem();

    VehicleManager vm;
    vm.move(vrp, 1);

    node.expand(vrp, vm);
    LONGS_EQUAL(5, node.childSize());
}

TEST(Node, expandWhenLastVehicleRun)
{
    Vrp_SetProblem();

    VehicleManager vm;
    vm.move(vrp, VehicleManager::CHANGE);

    node.expand(vrp, vm);
    LONGS_EQUAL(5, node.childSize());
}

TEST(Node, expandWhenRunningVehicleCapacityFull)
{
    Vrp_SetProblem();

    VehicleManager vm;

    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    node.expand(vrp, vm);
    LONGS_EQUAL(1, node.childSize());
}

TEST(Node, doNotExpand)
{
    Vrp_SetProblem();

    VehicleManager vm;

    vm.move(vrp, VehicleManager::CHANGE);
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    node.expand(vrp, vm);
    LONGS_EQUAL(0, node.childSize());
}

TEST(Node, nodeExpandWhenNodeSearch)
{
    Vrp_SetProblem();

    VehicleManager vm;

    node.search(vrp, vm);

    LONGS_EQUAL(6, node.childSize());
}

TEST(Node, valueIsAddedWhenNodeSearch)
{
    Vrp_SetProblem();

    VehicleManager vm;

    srand(2013);
    node.search(vrp, vm);

    /* searchにより次の手を車体の変更としているためINF */
    LONGS_EQUAL(202, node.value());

    node.search(vrp, vm);

    LONGS_EQUAL(INF+202, node.value());
}

TEST(Node, searchOnce)
{
    Vrp_SetProblem();

    VehicleManager vm;

    node.search(vrp, vm);

    LONGS_EQUAL(1, node.count());
}

TEST(Node, searchTwice)
{
    Vrp_SetProblem();

    VehicleManager vm;

    node.search(vrp, vm);
    node.search(vrp, vm);

    LONGS_EQUAL(2, node.count());
}

TEST(Node, finishCheck)
{
    Vrp_SetProblem();

    VehicleManager vm;

    vm.move(vrp, VehicleManager::CHANGE);

    Node mct;
    /* 21というマジックナンバーは実験によって得たもの */
    for (int i=0; i < 21; i++)
        mct.search(vrp, vm);
}
