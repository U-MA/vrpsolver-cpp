#ifndef VRPSOLVER_CPP_MCT_SELECTOR_H
#define VRPSOLVER_CPP_MCT_SELECTOR_H

#include <vector>

#include "mct_node.h"

namespace Selector
{
    MctNode *Ucb(MctNode& root, std::vector<MctNode*>& visited);
}

#endif /* VRPSOLVER_CPP_MCT_SELECTOR_H */
