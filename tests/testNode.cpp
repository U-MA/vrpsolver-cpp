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

TEST(Node, selectChildWhenNodeDontHaveChild)
{
    Node *selected = node.select();
    POINTERS_EQUAL(NULL, selected);
}

TEST(Node, selectChild)
{
    node.expand(1);
    Node *selected = node.select();
    CHECK(selected != NULL);
}
