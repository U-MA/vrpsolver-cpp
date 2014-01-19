#include "CppUTest/TestHarness.h"

#include <string.h>

#include "VrpProblems.h"

#include "solver.h"

TEST_GROUP(Solver)
{
};

TEST(Solver, init)
{
    vrp_problem *vrp = VrpProblem::E_n13_k4();
    VrpProblem::teardown(vrp);
}

TEST(Solver, init2)
{
    vrp_problem *vrp = VrpProblem::E_n51_k5();
    VrpProblem::teardown(vrp);
}

TEST(Solver, init3)
{
    vrp_problem *vrp = VrpProblem::E_n101_k14();
    VrpProblem::teardown(vrp);
}
