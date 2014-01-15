#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "wrapper_vrp.h"

TEST_GROUP(WrapperVrp)
{
};

TEST(WrapperVrp, fail)
{
    char infile[300] = "Vrp-All/E/E-n13-k4.vrp";
    vrp_problem *vrp = createVrpFromFilePath(infile);
    LONGS_EQUAL(4, vrp->numroutes);
    destroyVrp(vrp);
}
