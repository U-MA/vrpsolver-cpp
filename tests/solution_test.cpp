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

    void VisitAllCustomer()
    {
        solution->CurrentVehicle()->Visit(vrp, 1);

        solution->ChangeVehicle();
        solution->CurrentVehicle()->Visit(vrp, 8);
        solution->CurrentVehicle()->Visit(vrp, 5);
        solution->CurrentVehicle()->Visit(vrp, 3);

        solution->ChangeVehicle();
        solution->CurrentVehicle()->Visit(vrp, 9);
        solution->CurrentVehicle()->Visit(vrp, 12);
        solution->CurrentVehicle()->Visit(vrp, 10);
        solution->CurrentVehicle()->Visit(vrp, 6);

        solution->ChangeVehicle();
        solution->CurrentVehicle()->Visit(vrp, 11);
        solution->CurrentVehicle()->Visit(vrp, 4);
        solution->CurrentVehicle()->Visit(vrp, 7);
        solution->CurrentVehicle()->Visit(vrp, 2);
    }
};

TEST(Solution, Copy)
{
    Solution solution_copy(vrp);
    solution->Copy(solution_copy);
    CHECK(solution_copy.CurrentVehicle() != solution->CurrentVehicle());
}

TEST(Solution, IsFeasible)
{
    VisitAllCustomer();
    CHECK_TRUE(solution->IsFeasible());
}

TEST(Solution, IsNotFeasibleWhenCreate)
{
    CHECK_FALSE(solution->IsFeasible());
}

TEST(Solution, IsNotFeasibleWhenAllVehicleIsOnlyRun)
{
    solution->ChangeVehicle();
    solution->ChangeVehicle();
    solution->ChangeVehicle();
    solution->ChangeVehicle();
    CHECK_FALSE(solution->IsFeasible());
}

TEST(Solution, IsFinishWhenAllCustomerAreVisited)
{
    VisitAllCustomer();
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
    VisitAllCustomer();
    LONGS_EQUAL(247, solution->ComputeTotalCost(vrp));
}

TEST(Solution, IsVisit)
{
    solution->CurrentVehicle()->Visit(vrp, 2);
    CHECK_TRUE(solution->IsVisit(2));
}

TEST(Solution, FirstVehicleIs0)
{
    LONGS_EQUAL(0, solution->CurrentVehicleId());
}

TEST(Solution, SecontVehicleIs1)
{
    solution->ChangeVehicle();
    LONGS_EQUAL(1, solution->CurrentVehicleId());
}
