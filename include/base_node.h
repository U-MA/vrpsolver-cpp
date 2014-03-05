#ifndef VRPSOLVER_CPP_BASE_NODE_H
#define VRPSOLVER_CPP_BASE_NODE_H

#include "base_vrp.h"
#include "solution.h"

class BaseNode
{
public:
    virtual ~BaseNode() {}

    virtual unsigned int ChildSize() const = 0;
    virtual unsigned int Count() const = 0;
    virtual long Value() const = 0;

    virtual bool IsLeaf() const = 0;
    virtual void Expand(const BaseVrp& vrp, const Solution& solution) = 0;
    virtual void Update(long value) = 0;
};

#endif /* VRPSOLVER_CPP_BASE_NODE_H */
