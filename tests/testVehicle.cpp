#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "Vehicle.h"


TEST_GROUP(Vehicle)
{
    vrp_problem *vrp;
    Vehicle v;

    void setup()
    {
        vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
        vrp->demand = (int *)calloc(10, sizeof(int));
        vrp->dist.cost = (int *)calloc(10, sizeof(int));

        vrp->vertnum = 10;
        vrp->dist.cost[0] = 20; /* 0-1 */
        vrp->dist.cost[1] = 30; /* 0-2 */
        vrp->dist.cost[2] = 15; /* 1-2 */

        vrp->demand[1] = 1900;
        vrp->demand[2] = 1100;
        vrp->demand[9] = 900;
   }

    void teardown()
    {
        free(vrp->dist.cost);
        free(vrp->demand);
        free(vrp);
    }
};

TEST(Vehicle, init)
{
    LONGS_EQUAL(0, v.quantity());
}

TEST(Vehicle, visit)
{
    v.visit(vrp, 1);
    LONGS_EQUAL(1900, v.quantity());

    v.visit(vrp, 2);
    LONGS_EQUAL(3000, v.quantity());
}

TEST(Vehicle, computeCost)
{
    LONGS_EQUAL(0, v.computeCost(vrp));

    v.visit(vrp, 1);
    LONGS_EQUAL(40, v.computeCost(vrp));

    v.visit(vrp, 2);
    LONGS_EQUAL(65, v.computeCost(vrp));
}

TEST(Vehicle, copy)
{
    v.visit(vrp, 1);
    Vehicle copy = v.copy();
    LONGS_EQUAL(v.quantity(), copy.quantity());
}

/* テスト方法がわからないので保留 */
/*
TEST(Vehicle, visitOverMaxSize)
{
}
*/
