#include <cmath>
#include <iostream>
#include <vector>

#include "mct_selector.h"


MctNode *Selector::Ucb(MctNode& root, std::vector<MctNode *>& visited)
{
    MctNode *node = &root;
    while (!node->IsLeaf())
    {
        unsigned int next = 0;
        for (unsigned int i=0; i < node->ChildSize(); i++)
        {
            double ucb, max_ucb = .0;
            MctNode *child = node->Child(i);
            ucb = (double)child->Value() / child->Count() +
                  100.0 * sqrt(log(node->Count()) / child->Count());
            if (ucb > max_ucb)
            {
                max_ucb = ucb;
                next = i;
            }
        }
        node = node->Child(next);
    }
    return node;
}
