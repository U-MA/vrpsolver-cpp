#ifndef VRPSOLVER_CPP_BASE_NODE_H
#define VRPSOLVER_CPP_BASE_NODE_H

#include "base_vrp.h"
#include "solution.h"

class BaseNode
{
public:
    virtual ~BaseNode() {}

    virtual int CustomerId() const = 0;
    virtual int Count()      const = 0;
    virtual int ChildSize()  const = 0;
    virtual int Value()      const = 0;

    virtual bool IsLeaf() const = 0;
    virtual void Expand(const BaseVrp& vrp, const Solution& solution) = 0;
    virtual void Update(int value) = 0;
};

#endif /* VRPSOLVER_CPP_BASE_NODE_H */
