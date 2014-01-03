#include "Node.h"

Node::Node(void)
{
    customer_  = 0;
    count_     = 0;
    childSize_ = 0;
    value_     = 0;
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

int Node::value(void) const
{
    return value_;
}

void Node::expand(int childSize)
{
    childSize_ = childSize;
    child = new Node[childSize];
    for (int i=0; i < childSize_; i++)
    {
        child[i].customer_ = i;
    }
}

double Node::computeUcb(void)
{
    double ucb = 1e6;
    if (count_ != 0)
        ucb = value_ / count_;

    return ucb;
}

Node *Node::select(void)
{
    Node *selected = NULL;
    double maxUcb = -1.0;
    for (int i=0; i < childSize_; i++)
    {
        double ucb = child[i].computeUcb();
        if (ucb > maxUcb)
        {
            maxUcb = ucb;
            selected = &child[i];
        }
    }

    return selected;
}

bool Node::isLeaf(void) const
{
    return (childSize_ == 0);
}

void Node::update(int value)
{
    count_++;
    value_ += value;
}
