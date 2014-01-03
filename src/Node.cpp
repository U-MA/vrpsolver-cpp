#include "Node.h"

Node::Node(void)
{
    count_     = 0;
    childSize_ = 0;
}

Node::~Node(void)
{
    if (childSize_ != 0)
        delete[] child;
}

int Node::count(void) const
{
    return count_;
}

int Node::childSize(void) const
{
    return childSize_;
}

void Node::expand(int childSize)
{
    childSize_ = childSize;
    child = new Node[childSize];
}

Node *Node::select(void)
{
    if (childSize_ == 0)
        return NULL;
    else
        return child;
}
