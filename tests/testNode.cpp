#include "CppUTest/TestHarness.h"

#include "Node.h"


TEST_GROUP(Node)
{
};

TEST(Node, start)
{
    Node node;
    LONGS_EQUAL(0, node.count());
}
