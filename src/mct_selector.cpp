#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "mct_selector.h"


static double CalcUcb(MctNode *parent, MctNode *child, double coef)
{
    if (child->Count() == 0)
        return 100000 + (rand() % 100000);

    return ((double)child->Value() / child->Count()) +
           coef * sqrt(log(parent->Count()) / child->Count());
}

MctNode *Selector::Ucb(MctNode& root, std::vector<MctNode *>& visited, double coef)
{
    MctNode *node = &root;
    visited.push_back(node);
    while (!node->IsLeaf())
    {
        unsigned int next = 0;
        double max_ucb = -10000000;
        for (unsigned int i=0; i < node->ChildSize(); i++)
        {
            double ucb = CalcUcb(node, node->Child(i), coef);
            if (ucb > max_ucb)
            {
                max_ucb = ucb;
                next = i;
            }
        }
        node = node->Child(next);
        visited.push_back(node);
    }
    return node;
}
