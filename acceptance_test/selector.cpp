#include "CppUTest/TestHarness.h"

#include <vector>

#include "mct_node.h"
#include "mct_selector.h"

using namespace std;

TEST_GROUP(SelectorAcceptance)
{
    vector<MctNode *> visited;
};

TEST(SelectorAcceptance, Myself)
{
    MctNode node(0);

    MctNode *same_node;
    same_node = Selector::Ucb(node, visited);

    POINTERS_EQUAL(&node, same_node);
    POINTERS_EQUAL(&node, visited[0]);
}

TEST(SelectorAcceptance, TreeDepth2)
{
    MctNode node(0);
    node.CreateChild(0);
    node.CreateChild(1);

    MctNode *child0;
    child0 = Selector::Ucb(node, visited);

    POINTERS_EQUAL(node.Child(0), child0);
    POINTERS_EQUAL(&node,         visited[0]);
    POINTERS_EQUAL(node.Child(0), visited[1]);
}

TEST(SelectorAcceptance, SelectChild1)
{
    MctNode node(0);
    node.CreateChild(0);
    node.CreateChild(1);
    node.Child(0)->Update(100);
    node.Child(1)->Update(200);
    node.Update(100);
    node.Update(200);

    MctNode *child1;
    child1 = Selector::Ucb(node, visited);

    POINTERS_EQUAL(node.Child(1), child1);
    POINTERS_EQUAL(&node,         visited[0]);
    POINTERS_EQUAL(node.Child(1), visited[1]);
}
