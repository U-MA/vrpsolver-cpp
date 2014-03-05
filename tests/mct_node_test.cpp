#include "CppUTest/TestHarness.h"

#include <climits>

#include "mct_node.h"

TEST_GROUP(MctNode)
{
};

TEST(MctNode, Initialize)
{
    MctNode node(1);
    LONGS_EQUAL(0, node.Count());
    LONGS_EQUAL(1, node.CustomerId());
    LONGS_EQUAL(0, node.Value());
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

/* 例外処理についてまだ未勉強なので保留
TEST(MctNode, UpdateOverFlow)
{
    MctNode node(1);
    node.Update(LONG_MAX);
    CHECK_FALSE(node.Update(1));
}
*/
