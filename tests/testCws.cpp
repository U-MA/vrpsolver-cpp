#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
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

    vrp->dist.cost[0] = 20; /* 0-1 */
    vrp->dist.cost[1] = 30; /* 0-2 */
    vrp->dist.cost[2] = 15; /* 1-2 */

    Savings s;
    s.set(vrp, 1, 2);
    LONGS_EQUAL(35, s.getValue());
    LONGS_EQUAL(1, s.getEdge().first);
    LONGS_EQUAL(2, s.getEdge().second);
}

TEST(Cws, savingsLessThan)
{
    vrp->vertnum = 4;
    vrp->edgenum = 6;

    vrp->dist.cost[0] = 20; /* 0-1 */
    vrp->dist.cost[1] = 30; /* 0-2 */
    vrp->dist.cost[2] = 15; /* 1-2 */
    vrp->dist.cost[3] = 40; /* 0-3 */
    vrp->dist.cost[4] = 10; /* 1-3 */
    vrp->dist.cost[5] = 30; /* 2-3 */

    Savings s1, s2;
    s1.set(vrp, 1, 2); /* 35 */
    s2.set(vrp, 2, 3); /* 40 */

    CHECK_TRUE(s1 < s2);
}


TEST(Cws, InitSavingsList)
{
    SavingsList sl(NULL);
    LONGS_EQUAL(0, sl.getSize());
}

TEST(Cws, SetSizeFromVrpData)
{
    vrp->edgenum = 30;

    SavingsList sl(vrp);
    LONGS_EQUAL(30, sl.getSize());
}

TEST(Cws, getEdgeFromSavingsList)
{
    vrp->edgenum = 3;
    vrp->vertnum = 3;

    vrp->dist.cost[0] = 20;
    vrp->dist.cost[1] = 30;
    vrp->dist.cost[2] = 15;

    SavingsList sl(vrp);
    LONGS_EQUAL(1, sl.getEdge().first);
    LONGS_EQUAL(2, sl.getEdge().second);
}
