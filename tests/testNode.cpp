#include "CppUTest/TestHarness.h"

#include "Node.h"


TEST_GROUP(Node)
{
    Node node;
};

TEST(Node, create)
{
    LONGS_EQUAL(0, node.customer());
    LONGS_EQUAL(0, node.count());
    LONGS_EQUAL(0, node.childSize());
    LONGS_EQUAL(0, node.value());
}

TEST(Node, isLeaf)
{
    CHECK_TRUE(node.isLeaf());

    node.expand(1);
    CHECK_FALSE(node.isLeaf());
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
    LONGS_EQUAL(0, selected->customer());
}

TEST(Node, selectChildWithMaxUcb)
{
    node.expand(2);
    Node *selected = node.select();
    selected->update(100);
    selected = node.select();
    LONGS_EQUAL(1, selected->customer());
}

TEST(Node, update)
{
    node.expand(1);
    Node *selected = node.select();
    selected->update(100);
    LONGS_EQUAL(1, selected->count());
    LONGS_EQUAL(100, selected->value());
    selected->update(30);
    LONGS_EQUAL(2, selected->count());
    LONGS_EQUAL(130, selected->value());
}
