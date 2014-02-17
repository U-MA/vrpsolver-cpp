#include "CppUTest/TestHarness.h"

#include "host_vrp.h"
#include "solution.h"

TEST_GROUP(Solution)
{
};

TEST(Solution, IsNotFinishWhenCreate)
{
    HostVrp vrp;
    Solution solution(vrp);
    CHECK_FALSE(solution.IsFinish());
}
