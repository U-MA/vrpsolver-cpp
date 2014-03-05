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

TEST(MctNode, Update)
{
    MctNode node(1);
    node.Update(10);
    LONGS_EQUAL(1 , node.Count());
    LONGS_EQUAL(10, node.Value());
}
