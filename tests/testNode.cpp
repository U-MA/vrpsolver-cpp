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

TEST(Node, createChild)
{
    node.expand(5);
    LONGS_EQUAL(5, node.childSize());
}
