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
    LONGS_EQUAL(0, vm.getSize());
}

TEST(FixVehicleManager, addVehicle)
{
    VehicleManager vm;
    Vehicle v;
    vm.add(v);
    CHECK_FALSE(vm.empty());
    LONGS_EQUAL(1, vm.getSize());
}
