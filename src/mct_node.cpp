#include "mct_node.h"

bool MctNode::IsLeaf() const
{
    return false;
}

void MctNode::Expand(const BaseVrp& vrp, const Solution& solution)
{
}

void MctNode::Update(int value)
{
    count_++;
    value_ += value;
}

