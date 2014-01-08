#include "CppUTest/TestHarness.h"

#include "Solver.h"

TEST_GROUP(Solver)
{
};

TEST(Solver, setSeed)
{
    Solver::setSeed(2013);
    LONGS_EQUAL(33832491, rand());
}
