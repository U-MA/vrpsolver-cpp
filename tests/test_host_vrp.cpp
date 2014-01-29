#include "CppUTest/TestHarness.h"

#include "host_vrp.h"

TEST_GROUP(hostVrp)
{
    HostVrp *hvrp;

    void setup()
    {
        hvrp = new HostVrp;
    }

    void teardown()
    {
        delete hvrp;
    }
};

TEST(hostVrp, functionCheck)
{
    hvrp->Create("Vrp-All/E/E-n13-k4.vrp");

    LONGS_EQUAL(12,   hvrp->customer_size());
    LONGS_EQUAL(4,    hvrp->vehicle_size());
    LONGS_EQUAL(6000, hvrp->capacity());
    LONGS_EQUAL(9,    hvrp->cost(0, 1));
    LONGS_EQUAL(10,   hvrp->cost(11, 12));
    LONGS_EQUAL(1200, hvrp->demand(1));
    LONGS_EQUAL(1200, hvrp->demand(7));
    LONGS_EQUAL(1100, hvrp->demand(12));
}
