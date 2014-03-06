#include "mct_node.h"

MctNode::~MctNode()
{
    for (unsigned int i=0; i < child_size_; i++)
        delete(child_[i]);
}

bool MctNode::IsLeaf() const
{
    return false;
}

void MctNode::Expand(const BaseVrp& vrp, const Solution& solution)
{
}

void MctNode::Update(long value)
{
    count_++;
    value_ += value;
}

void MctNode::CreateChild(int customer_id)
{
    child_[child_size_] = new MctNode(customer_id);
    child_size_++;
}
