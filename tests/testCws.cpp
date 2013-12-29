#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "Cws.h"

TEST_GROUP(Cws)
{
};

TEST(Cws, InitSavings)
{
    Savings s;
    LONGS_EQUAL(0, s.getValue());
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().first);
    LONGS_EQUAL(Savings::UNKNOWN, s.getEdge().second);
}

TEST(Cws, InitSavingsList)
{
    SavingsList sl(NULL);
    LONGS_EQUAL(0, sl.getSize());
}

TEST(Cws, SetUpFromVrpData)
{
    vrp_problem *vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
    vrp->edgenum = 30;

    SavingsList sl(vrp);
    LONGS_EQUAL(30, sl.getSize());

    free(vrp);
}
