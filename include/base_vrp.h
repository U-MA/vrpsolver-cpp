#ifndef VRPSOLVER_CPP_BASE_VRP_H
#define VRPSOLVER_CPP_BASE_VRP_H

class BaseVrp
{
public:
    virtual ~BaseVrp(void) {};

    virtual int customer_size(void)  const = 0;
    virtual int vehicle_size(void)   const = 0;
    virtual int capacity(void)       const = 0;
    virtual int cost(int v0, int v1) const = 0;
    virtual int demand(int v)        const = 0;
};

#endif /* VRPSOLVER_CPP_BASE_VRP_H */
