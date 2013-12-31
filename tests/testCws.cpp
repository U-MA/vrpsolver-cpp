#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "Cws.h"

TEST_GROUP(Cws)
{
    vrp_problem *vrp;

    void setup()
    {
        vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
        vrp->dist.cost = (int *)calloc(100, sizeof(int));
    }

    void teardown()
    {
        free(vrp->dist.cost);
        free(vrp);
    }

    void Vrp_SetCost(int first, int second, int value)
    {
        vrp->dist.cost[INDEX(first, second)] = value;
    }
};

TEST(Cws, InitSavings)
{
    Savings s;
    LONGS_EQUAL(0, s.getValue());
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().first);
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().second);
}

TEST(Cws, setSavingsValue)
{
    vrp->edgenum = 3;
    vrp->vertnum = 3;

    Vrp_SetCost(0, 1, 20);
    Vrp_SetCost(0, 2, 30);
    Vrp_SetCost(1, 2, 15);

    Savings s;
    s.set(vrp, 1, 2);
    LONGS_EQUAL(35, s.getValue());
    LONGS_EQUAL(1, s.getEdge().first);
    LONGS_EQUAL(2, s.getEdge().second);
}

TEST(Cws, savingsCompare)
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
    s1.set(vrp, 1, 2); /* 35 */
    s2.set(vrp, 2, 3); /* 40 */

    CHECK_TRUE(s1 < s2);
    CHECK_FALSE(s1 > s2);
}

TEST(Cws, savingsCompareEqual)
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


TEST(Cws, SetSizeFromVrpData)
{
    vrp->vertnum = 100;

    SavingsList sl(vrp);
    LONGS_EQUAL(4851, sl.getSize());
}

TEST(Cws, getEdgeFromSavingsList)
{
    vrp->edgenum = 3;
    vrp->vertnum = 3;

    Vrp_SetCost(0, 1, 20);
    Vrp_SetCost(0, 2, 30);
    Vrp_SetCost(1, 2, 15);

    SavingsList sl(vrp);
    LONGS_EQUAL(1, sl.getEdge().first);
    LONGS_EQUAL(2, sl.getEdge().second);
}
