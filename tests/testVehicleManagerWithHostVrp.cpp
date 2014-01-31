#include "CppUTest/TestHarness.h"

#include "host_vrp.h"
#include "vehicle_manager.h"

TEST_GROUP(VehicleManagerWithHostVrp)
{
    HostVrp host_vrp;

    void setup()
    {
        host_vrp.Create("Vrp-All/E/E-n13-k4.vrp");
    }
};

TEST(VehicleManagerWithHostVrp, isVisitAll)
{
    VehicleManager vm;
    CHECK_FALSE(vm.isVisitAll(host_vrp));
}
