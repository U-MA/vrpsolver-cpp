#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "VrpProblems.h"

#include "SavingsList.h"

TEST_GROUP(SavingsList)
{
    vrp_problem *vrp;

    void setup()
    {
        vrp = VrpProblem::AMCT2CVRP();
    }

    void teardown()
    {
        VrpProblem::teardown(vrp);
    }
};

TEST(SavingsList, InitSavings)
{
    Savings s;
    LONGS_EQUAL(0, s.getValue());
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().first);
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().second);
}

TEST(SavingsList, setSavingsValue)
{
    

    Savings s;
    s.set(vrp, 1, 5);
    LONGS_EQUAL(42, s.getValue());
    LONGS_EQUAL(1, s.getEdge().first);
    LONGS_EQUAL(5, s.getEdge().second);
}

/*
TEST(SavingsList, savingsCompare)
{
    vrp->vertnum = 4;
    vrp->edgenum = 6;

    Vrp_SetCost(0, 1, 20);
    Vrp_SetCost(0, 2, 30);
    Vrp_SetCost(1, 2, 15);
    Vrp_SetCost(0, 3, 40);
    Vrp_SetCost(1, 3, 10);
    Vrp_SetCost(2, 3, 30);

    Savings s1, s2;
    s1.set(vrp, 1, 2); 
    s2.set(vrp, 2, 3);

    CHECK_TRUE(s1 < s2);
    CHECK_FALSE(s1 > s2);
}

TEST(SavingsList, savingsCompareEqual)
{
    vrp->vertnum = 3;
    vrp->edgenum = 3;

    Vrp_SetCost(0, 1, 20);
    Vrp_SetCost(0, 2, 30);
    Vrp_SetCost(1, 2, 15);
    Vrp_SetCost(0, 3, 40);
    Vrp_SetCost(1, 3, 10);
    Vrp_SetCost(2, 3, 30);

    Savings s1, s2, s3;
    s1.set(vrp, 1, 2);
    s2.set(vrp, 1, 2);
    s3.set(vrp, 2, 3);

    CHECK_TRUE(s1 <= s2);
    CHECK_TRUE(s1 <= s3);

    CHECK_TRUE(s1 >= s2);
    CHECK_FALSE(s1 >= s3);
}
*/


TEST(SavingsList, SetSizeFromVrpData)
{
    vrp->vertnum = 100;

    SavingsList sl(vrp);
    LONGS_EQUAL(4851, sl.getSize());
}

TEST(SavingsList, getEdgeFromSavingsList)
{
    

    SavingsList sl(vrp);
    EDGE edge = sl.getEdge();
    LONGS_EQUAL(1, edge.first);
    LONGS_EQUAL(5, edge.second);
    edge = sl.getEdge();
    LONGS_EQUAL(1, edge.first);
    LONGS_EQUAL(2, edge.second);
}
