#ifndef VRPSOLVER_CPP_MCT_NODE_H
#define VRPSOLVER_CPP_MCT_NODE_H

#include "base_node.h"

class MctNode : public BaseNode
{
public:
    MctNode(int customer_id) : child_(),
                child_size_(),
                count_(0),
                customer_id_(customer_id),
                value_(0)
    {
        if (customer_id < 0) customer_id_ = 0;
    }

    ~MctNode();

    MctNode* Child(unsigned int child_id) const;

    unsigned int ChildSize() const { return child_size_; }

    unsigned int Count() const { return count_; }

    unsigned int CustomerId() const { return customer_id_; }

    long Value() const { return value_; }

    void CreateChild(unsigned int customer_id);
    bool IsLeaf() const;
    void Expand(const BaseVrp& vrp, const Solution& solution);
    void Update(long value);

private:
    static const int kMaxChildSize = 120;

    MctNode* child_[kMaxChildSize];
    unsigned int child_size_;
    unsigned int count_;
    unsigned int customer_id_;
    long value_;
};

#endif /* VRPSOLVER_CPP_MCT_NODE_H */
