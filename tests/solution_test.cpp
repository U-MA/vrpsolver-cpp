#include "CppUTest/TestHarness.h"

#include "host_vrp.h"
#include "solution.h"
#include "vehicle.h"

TEST_GROUP(Solution)
{
    HostVrp vrp;

    void setup()
    {
        vrp.Create("./Vrp-All/E/E-n13-k4.vrp");
    }
};

TEST(Solution, IsFinish)
{
    Solution solution(vrp);

    solution.current_vehicle()->visit(vrp, 1);

    solution.ChangeVehicle();
    solution.current_vehicle()->visit(vrp, 8);
    solution.current_vehicle()->visit(vrp, 5);
    solution.current_vehicle()->visit(vrp, 3);

    solution.ChangeVehicle();
    solution.current_vehicle()->visit(vrp, 9);
    solution.current_vehicle()->visit(vrp, 12);
    solution.current_vehicle()->visit(vrp, 10);
    solution.current_vehicle()->visit(vrp, 6);

    solution.ChangeVehicle();
    solution.current_vehicle()->visit(vrp, 11);
    solution.current_vehicle()->visit(vrp, 4);
    solution.current_vehicle()->visit(vrp, 7);
    solution.current_vehicle()->visit(vrp, 2);

    CHECK_TRUE(solution.IsFinish());
}

TEST(Solution, IsNotFinishWhenCreate)
{
    Solution solution(vrp);
    CHECK_FALSE(solution.IsFinish());
}
