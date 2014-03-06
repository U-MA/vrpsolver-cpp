#include "mct_selector.h"

#include <vector>

MctNode *Selector::Ucb(MctNode& root, std::vector<MctNode *>& visited)
{
    MctNode *node = &root;
    while (!node->IsLeaf())
    {
        node = node->Child(0);
    }
    return node;
}
