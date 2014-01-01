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

        v = Vehicle(vrp->vertnum);
   }

    void teardown()
    {
        free(vrp->dist.cost);
        free(vrp->demand);
        free(vrp);
    }
};

TEST(Vehicle, visitOutOfCustomer)
{
    CHECK_FALSE(v.visit(vrp, 0));
    CHECK_FALSE(v.visit(vrp, 10));
}

TEST(Vehicle, visitInCustomer)
{
    CHECK_TRUE(v.visit(vrp, 1));
    CHECK_TRUE(v.visit(vrp, 9));
}

TEST(Vehicle, isNotVisit)
{
    CHECK_EQUAL(0, v.getQuantity());
}

TEST(Vehicle, visit)
{
    v.visit(vrp, 1);
    LONGS_EQUAL(1, v.getRoute(0));
    LONGS_EQUAL(1900, v.getQuantity());

    v.visit(vrp, 2);
    LONGS_EQUAL(3000, v.getQuantity());
}

TEST(Vehicle, getRouteFromOutOfCustomer)
{
    CHECK_EQUAL(OUT_OF_BOUND, v.getRoute(-1));
}

TEST(Vehicle, computeCost)
{
    LONGS_EQUAL(0, v.computeCost(vrp));

    v.visit(vrp, 1);
    LONGS_EQUAL(40, v.computeCost(vrp));

    v.visit(vrp, 2);
    LONGS_EQUAL(65, v.computeCost(vrp));
}

TEST(Vehicle, empty)
{
    CHECK_TRUE(v.empty());
}

TEST(Vehicle, notEmpty)
{
    v.visit(vrp, 1);
    CHECK_FALSE(v.empty());
}

/* テスト方法がわからないので保留 */
/*
TEST(Vehicle, visitOverMaxSize)
{
}
*/
