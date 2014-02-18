#include "CppUTest/TestHarness.h"

#include "host_vrp.h"
#include "solution.h"

TEST_GROUP(Solution)
{
    HostVrp  vrp;
    Solution *solution;

    void setup()
    {
        vrp.Create("./Vrp-All/E/E-n13-k4.vrp");
        solution = new Solution(vrp);
    }

    void teardown()
    {
        delete solution;
    }

    void VisitAllCustomer(const BaseVrp& vrp, Solution *solution)
    {
        solution->current_vehicle()->visit(vrp, 1);

        solution->ChangeVehicle();
        solution->current_vehicle()->visit(vrp, 8);
        solution->current_vehicle()->visit(vrp, 5);
        solution->current_vehicle()->visit(vrp, 3);

        solution->ChangeVehicle();
        solution->current_vehicle()->visit(vrp, 9);
        solution->current_vehicle()->visit(vrp, 12);
        solution->current_vehicle()->visit(vrp, 10);
        solution->current_vehicle()->visit(vrp, 6);

        solution->ChangeVehicle();
        solution->current_vehicle()->visit(vrp, 11);
        solution->current_vehicle()->visit(vrp, 4);
        solution->current_vehicle()->visit(vrp, 7);
        solution->current_vehicle()->visit(vrp, 2);
    }
};

TEST(Solution, Copy)
{
    Solution solution_copy(vrp);
    solution->Copy(solution_copy);
    CHECK(solution_copy.current_vehicle() != solution->current_vehicle());
}

TEST(Solution, IsFeasible)
{
    VisitAllCustomer(vrp, solution);
    CHECK_TRUE(solution->IsFeasible());
}

TEST(Solution, IsNotFeasibleWhenCreate)
{
    CHECK_FALSE(solution->IsFeasible());
}

TEST(Solution, IsFinishWhenAllCustomerAreVisited)
{
    VisitAllCustomer(vrp, solution);
    CHECK_TRUE(solution->IsFinish());
}

TEST(Solution, IsFinishWhenVehicleIsOver)
{
    for (int i=0; i < 5; i++)
        solution->ChangeVehicle();

    CHECK_TRUE(solution->IsFinish());
}

TEST(Solution, IsNotFinishWhenCreate)
{
    CHECK_FALSE(solution->IsFinish());
}

TEST(Solution, ComputeTotalCost)
{
    VisitAllCustomer(vrp, solution);
    LONGS_EQUAL(247, solution->ComputeTotalCost(vrp));
}

TEST(Solution, IsVisit)
{
    solution->current_vehicle()->visit(vrp, 2);
    CHECK_TRUE(solution->IsVisit(2));
}
