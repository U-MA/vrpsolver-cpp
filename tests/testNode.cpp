#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "VrpProblems.h"

#include "Node.h"
#include "VehicleManager.h"
#include "VrpSimulation.h"


TEST_GROUP(Node)
{
    vrp_problem *vrp;
    VehicleManager vm;
    Node node;
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
}

TEST(Node, isNotLeaf)
{
    node.expand(vrp, vm);
    CHECK_FALSE(node.isLeaf());
}

TEST(Node, createChild)
{
    node.expand(vrp, vm);
    LONGS_EQUAL(6, node.childSize()); /* 顧客数5 + depotの数1 */
}

TEST(Node, returnNullWhenNodeSelectBeforeExpand)
{
    Node *selected = node.select();
    POINTERS_EQUAL(NULL, selected);
}

TEST(Node, selectChild)
{
    node.expand(vrp, vm);
    Node *selected = node.select();
    LONGS_EQUAL(1, selected->customer());
}

TEST(Node, selectChildWithMaxUcb)
{
    node.expand(vrp, vm);
    Node *selected = node.select();
    selected->update(100);
    selected = node.select();
    LONGS_EQUAL(0, selected->customer());
}

TEST(Node, update)
{
    node.expand(vrp, vm);
    Node *selected = node.select();

    selected->update(100);
    LONGS_EQUAL(1, selected->count());
    LONGS_EQUAL(100, selected->value());
}

TEST(Node, expandWithVehicleManager)
{

    VehicleManager vm;

    node.expand(vrp, vm);
    LONGS_EQUAL(6, node.childSize());
}

TEST(Node, expandAfterVehicleVisitOneCustomer)
{
    VehicleManager vm;
    vm.move(vrp, 1);

    node.expand(vrp, vm);
    LONGS_EQUAL(5, node.childSize());
}

TEST(Node, expandWhenLastVehicleRun)
{
    VehicleManager vm;
    vm.move(vrp, VehicleManager::kChange);

    node.expand(vrp, vm);
    LONGS_EQUAL(5, node.childSize());
}

TEST(Node, expandWhenRunningVehicleCapacityFull)
{
    VehicleManager vm;

    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    node.expand(vrp, vm);
    LONGS_EQUAL(1, node.childSize());
}

TEST(Node, doNotExpand)
{
    VehicleManager vm;

    vm.move(vrp, VehicleManager::kChange);
    vm.move(vrp, 1);
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    node.expand(vrp, vm);
    LONGS_EQUAL(0, node.childSize());
}

TEST(Node, nodeExpandWhenNodeSearch)
{
    VehicleManager vm;

    node.search(vrp, vm, 1);

    LONGS_EQUAL(6, node.childSize());
}

TEST(Node, valueIsAddedWhenNodeSearch)
{
    VehicleManager vm;

    node.search(vrp, vm, 1);

    /* searchにより次の手を車体の変更としているためINF */
    LONGS_EQUAL(202, node.value());

    node.search(vrp, vm, 1);

    LONGS_EQUAL(373, node.value());
}

TEST(Node, valueIsAddedWhenNodeSearch2)
{
    VehicleManager vm;

    node.search(vrp, vm, 1);

    LONGS_EQUAL(202, node.value());
}

TEST(Node, searchOnce)
{

    VehicleManager vm;

    node.search(vrp, vm, 1);

    LONGS_EQUAL(1, node.count());
}

TEST(Node, searchTwice)
{
    VehicleManager vm;

    node.search(vrp, vm, 1);
    node.search(vrp, vm, 1);

    LONGS_EQUAL(2, node.count());
}

TEST(Node, DoPrunning)
{
    VehicleManager vm;

    Node mct;
    for (int i=0; i < 100; i++)
        mct.search(vrp, vm, 1);

    CHECK(mct.value() < INF);
}

TEST(Node, setTabu)
{
    VehicleManager vm;

    Node node;

    node.expand(vrp, vm);
    node.addTabu(3);
    CHECK_TRUE(node.tabu(3));
    CHECK_FALSE(node.tabu(1));
}

TEST(Node, isTabu)
{
    VehicleManager vm;

    Node node;
    node.expand(vrp, vm);
    for (int i=0; i < vrp->vertnum; i++)
        node.addTabu(i);
    CHECK_TRUE(node.isTabu(vrp));
}

TEST(Node, isNotTabu)
{
    VehicleManager vm;

    Node node;
    node.expand(vrp, vm);
    CHECK_FALSE(node.isTabu(vrp));
}

TEST(Node, isTabuInMiddle)
{
    VehicleManager vm;
    vm.move(vrp, 2);
    vm.move(vrp, 4);

    Node node;
    node.expand(vrp, vm);
    CHECK_FALSE(node.isTabu(vrp));
    node.addTabu(0);
    node.addTabu(1);
    node.addTabu(3);
    node.addTabu(5);
    CHECK_TRUE(node.isTabu(vrp));
}

TEST(Node, error)
{
    VrpProblem::teardown(vrp);
    vrp = VrpProblem::E_n13_k4();

    VehicleManager vm;

    Node mct;
    for (int i=0; i < 40; i++)
        mct.search(vrp, vm, 1);
}
