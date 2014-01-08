#include "CppUTest/TestHarness.h"

#include <string.h>

#include "Solver.h"

TEST_GROUP(Solver)
{
};

IGNORE_TEST(Solver, init)
{
    char filename[200];
    strcpy(filename, "../vrpsolver-cpp_sample/Vrp-All/E/E-n13-k4.vrp");
    Solver::setProblem(filename);
}
