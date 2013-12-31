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

TEST(Cws, InitSavings)
{
    Savings s;
    LONGS_EQUAL(0, s.getValue());
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().first);
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().second);
}

TEST(Cws, setSavingsValue)
{
    Vrp_SetProblem();

    Savings s;
    s.set(vrp, 1, 5);
    LONGS_EQUAL(42, s.getValue());
    LONGS_EQUAL(1, s.getEdge().first);
    LONGS_EQUAL(5, s.getEdge().second);
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
    Vrp_SetProblem();

    SavingsList sl(vrp);
    LONGS_EQUAL(1, sl.getEdge().first);
    LONGS_EQUAL(5, sl.getEdge().second);
}
