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
}

TEST(Selector, ucb_selector_2)
{
    MctNode root(0);
    root.CreateChild(0);

    POINTERS_EQUAL(root.Child(0), Selector::Ucb(root, visited));
}
