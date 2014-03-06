#include "mct_selector.h"

#include <vector>

MctNode *Selector::Ucb(MctNode& root, std::vector<MctNode *>& visited)
{
    if (root.IsLeaf()) return &root;
    return NULL;
}
