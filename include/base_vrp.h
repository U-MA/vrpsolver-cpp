#ifndef VRPSOLVER_CPP_BASE_VRP_H
#define VRPSOLVER_CPP_BASE_VRP_H

class BaseVrp
{
public:
    virtual ~BaseVrp(void) {};

    int customer_size(void)  const = 0;
    int vehicle_size(void)   const = 0;
    int capacity(void)       const = 0;
    int cost(int v0, int v1) const = 0;
    int demand(int v)        const = 0;
};

#endif /* VRPSOLVER_CPP_BASE_VRP_H */
