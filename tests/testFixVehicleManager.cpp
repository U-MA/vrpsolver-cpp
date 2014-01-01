#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

TEST_GROUP(FixVehicleManager)
{
    VehicleManager vm;
    Vehicle v;
};

TEST(FixVehicleManager, empty)
{
    CHECK_TRUE(vm.empty());
}

TEST(FixVehicleManager, addVehicle)
{
    vm.add(v);
    CHECK_FALSE(vm.empty());
}

TEST(FixVehicleManager, getSize)
{
    LONGS_EQUAL(0, vm.getSize());
    vm.add(v);
    LONGS_EQUAL(1, vm.getSize());
    vm.add(v);
    LONGS_EQUAL(2, vm.getSize());
}
