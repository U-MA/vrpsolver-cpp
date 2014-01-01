#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

TEST_GROUP(FixVehicleManager)
{
};

TEST(FixVehicleManager, empty)
{
    VehicleManager vm;
    CHECK_TRUE(vm.empty());
}

TEST(FixVehicleManager, addVehicle)
{
    VehicleManager vm;
    Vehicle v;
    vm.add(v);
    CHECK_FALSE(vm.empty());
}

TEST(FixVehicleManager, getSize)
{
    VehicleManager vm;
    
    LONGS_EQUAL(0, vm.getSize());

    Vehicle vehicle;
    vm.add(vehicle);

    LONGS_EQUAL(1, vm.getSize());

    vm.add(vehicle);

    LONGS_EQUAL(2, vm.getSize());
}
