#include "Node.h"

Node::Node(void)
{
    count_     = 0;
    childSize_ = 0;
}

Node::~Node(void)
{
}

int Node::count(void)
{
    return count_;
}

int Node::childSize(void)
{
    return childSize_;
}
