#include "CppUTest/TestHarness.h"

#include "mct_node.h"

TEST_GROUP(MctNode)
{
};

TEST(MctNode, CountIs0AfterConstract)
{
    MctNode node;
    LONGS_EQUAL(0, node.Count());
}

TEST(MctNode, ValueIs0AfterConstract)
{
    MctNode node;
    LONGS_EQUAL(0, node.Value());
}
