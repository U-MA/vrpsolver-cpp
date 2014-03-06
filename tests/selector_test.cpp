#include "CppUTest/TestHarness.h"

#include "mct_node.h"
#include "mct_selector.h"

TEST_GROUP(Selector)
{
    std::vector<MctNode *> visited;
};

TEST(Selector, ucb_selector)
{
    MctNode root(0);

    POINTERS_EQUAL(&root, Selector::Ucb(root, visited));
    POINTERS_EQUAL(&root, visited[0]);
}

TEST(Selector, ucb_selector_2)
{
    MctNode root(0);
    root.CreateChild(0);

    POINTERS_EQUAL(root.Child(0), Selector::Ucb(root, visited));
    POINTERS_EQUAL(&root, visited[0]);
    POINTERS_EQUAL(root.Child(0), visited[1]);
}

TEST(Selector, ucb_selector_3)
{
    MctNode root(0);
    root.CreateChild(0);
    root.CreateChild(1);

    root.Update(10);
    root.Update(100);
    root.Child(0)->Update(10);
    root.Child(1)->Update(100);

    POINTERS_EQUAL(root.Child(1), Selector::Ucb(root, visited));
    POINTERS_EQUAL(&root, visited[0]);
    POINTERS_EQUAL(root.Child(1), visited[1]);
}

TEST(Selector, ucb_selector_4)
{
    MctNode root(0);
    root.CreateChild(0);
    MctNode *child = root.Child(0);
    child->CreateChild(0);

    root.Update(100);
    root.Update(100);
    root.Child(0)->Update(100);
    child->Child(0)->Update(100);

    POINTERS_EQUAL(child->Child(0), Selector::Ucb(root, visited));
    POINTERS_EQUAL(&root, visited[0]);
    POINTERS_EQUAL(child, visited[1]);
    POINTERS_EQUAL(child->Child(0), visited[2]);
}
