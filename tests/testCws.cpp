#include "CppUTest/TestHarness.h"

#include "Cws.h"

TEST_GROUP(Cws)
{
};

TEST(Cws, InitSavings)
{
    Savings s;
    LONGS_EQUAL(0, s.getValue());
    LONGS_EQUAL(Savings::UNKNOWN, s.getFirst());
    LONGS_EQUAL(Savings::UNKNOWN, s.getSecond());
}

TEST(Cws, InitSavingsList)
{
    SavingsList sl(NULL);
    LONGS_EQUAL(0, sl.getSize());
}
