#include "CppUTest/TestHarness.h"

#include "mct_node.h"
#include "mct_selector.h"

TEST_GROUP(Selector)
{
};

TEST(Selector, ucb_selector)
{
    MctNode root(0);
    std::vector<MctNode *> visited;

    POINTERS_EQUAL(&root, Selector::Ucb(root, visited));
}
