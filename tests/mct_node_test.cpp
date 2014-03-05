#include "CppUTest/TestHarness.h"

#include "mct_node.h"

TEST_GROUP(MctNode)
{
};

TEST(MctNode, Initialize)
{
    MctNode node(1);
    LONGS_EQUAL(0, node.Count());
    LONGS_EQUAL(0, node.Value());
    LONGS_EQUAL(1, node.CustomerId());
}

TEST(MctNode, ConstractWithMinusCustomerId)
{
    MctNode node(-1);
    LONGS_EQUAL(0, node.CustomerId());
}
