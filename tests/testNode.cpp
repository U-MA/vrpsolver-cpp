#include "CppUTest/TestHarness.h"

#include "Node.h"


TEST_GROUP(Node)
{
    Node node;
};

TEST(Node, init)
{
    LONGS_EQUAL(0, node.count());
    LONGS_EQUAL(0, node.childSize());
}
