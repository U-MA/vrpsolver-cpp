#include "CppUTest/TestHarness.h"

#include "mct_node.h"

TEST_GROUP(MctNode)
{
};

TEST(MctNode, Initialize)
{
    MctNode node;
    LONGS_EQUAL(0, node.Count());
    LONGS_EQUAL(0, node.Value());
}
