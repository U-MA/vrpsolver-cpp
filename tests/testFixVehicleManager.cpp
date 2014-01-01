#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

TEST_GROUP(FixVehicleManager)
{
};

TEST(FixVehicleManager, start)
{
    VehicleManager vm;
    CHECK_TRUE(vm.empty());
}
