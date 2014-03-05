#include "CppUTest/TestHarness.h"

#include "mct_node.h"

TEST_GROUP(MctNode)
{
};

TEST(MctNode, getCount)
{
    MctNode node;
    LONGS_EQUAL(0, node.Count());
}
