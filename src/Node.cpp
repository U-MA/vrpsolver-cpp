#include "Node.h"

Node::Node(void)
{
    customer_  = 0;
    count_     = 0;
    childSize_ = 0;
    child      = NULL;
}

Node::~Node(void)
{
    if (childSize_ != 0)
        delete[] child;
}

int Node::customer(void) const
{
    return customer_;
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

bool Node::isLeaf(void) const
{
    return (childSize_ == 0);
}
