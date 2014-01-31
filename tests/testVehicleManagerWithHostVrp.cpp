#include "CppUTest/TestHarness.h"

#include "host_vrp.h"
#include "vehicle_manager.h"

TEST_GROUP(VehicleManagerWithHostVrp)
{
    HostVrp host_vrp;
    VehicleManager vm;

    void setup()
    {
        host_vrp.Create("Vrp-All/E/E-n13-k4.vrp");
    }
};

TEST(VehicleManagerWithHostVrp, isNotVisitAll)
{
    CHECK_FALSE(vm.isVisitAll(host_vrp));
}

TEST(VehicleManagerWithHostVrp, isVisitAll)
{
    vm.move(host_vrp, 1);
    vm.changeVehicle();
    vm.move(host_vrp, 8);
    vm.move(host_vrp, 5);
    vm.move(host_vrp, 3);
    vm.changeVehicle();
    vm.move(host_vrp, 9);
    vm.move(host_vrp, 12);
    vm.move(host_vrp, 10);
    vm.move(host_vrp, 6);
    vm.changeVehicle();
    vm.move(host_vrp, 11);
    vm.move(host_vrp, 4);
    vm.move(host_vrp, 7);
    vm.move(host_vrp, 2);

    CHECK_TRUE(vm.isVisitAll(host_vrp));
}

TEST(VehicleManagerWithHostVrp, move)
{
    vm.move(host_vrp, 1);
    CHECK_TRUE(vm.isVisit(1));
    LONGS_EQUAL(18, vm.computeTotalCost(host_vrp));
}

