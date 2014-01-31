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
    v.visit(host_vrp, 1);
    LONGS_EQUAL(1200, v.capacity());
}
