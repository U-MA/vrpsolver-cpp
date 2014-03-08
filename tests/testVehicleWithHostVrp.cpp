#include "CppUTest/TestHarness.h"

#include "host_vrp.h"
#include "vehicle.h"

TEST_GROUP(VehicleWithHostVrp)
{
    HostVrp host_vrp;

    void setup()
    {
        host_vrp.Create("Vrp-All/E/E-n13-k4.vrp");
    }
};

TEST(VehicleWithHostVrp, visit)
{
    Vehicle v;
    v.Visit(host_vrp, 1);
    LONGS_EQUAL(1200, v.Capacity());
    v.Visit(host_vrp, 12);
    LONGS_EQUAL(2300, v.Capacity());
}

TEST(VehicleWithHostVrp, computeCost)
{
    Vehicle v;
    v.Visit(host_vrp, 1);
    LONGS_EQUAL(18, v.ComputeCost(host_vrp));
    v.Visit(host_vrp, 12);
    LONGS_EQUAL(47, v.ComputeCost(host_vrp));
}
