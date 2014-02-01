#ifndef VRPSOLVER_CPP_NODE_H
#define VRPSOLVER_CPP_NODE_H

#include <cstdlib>

#include "base_vrp.h"
#include "vehicle_manager.h"

class Node
{
public:
    Node(void) : customer_(0), count_(0), child_size_(0), value_(0),
                 child_(NULL), tabu_(NULL) {};
    ~Node(void);

    /* getter */
    int  customer(void)     const { return customer_; }
    int  count(void)        const { return count_; }
    int  childSize(void)    const { return child_size_; }
    int  value(void)        const { return value_; }
    bool tabu(int customer) const { return tabu_[customer]; }

    void addTabu(int customer);

    bool isLeaf(void) const;
    bool isTabu(const BaseVrp& vrp) const;
    void expand(const BaseVrp& vrp, const VehicleManager& vm);
    void update(int value);

    /* privateにしたいがテストのためpublicにしている */
    Node *selectMaxUcbChild(void);

    /* for MonteCarloTreeSearch */
    void build(const BaseVrp& vrp, const VehicleManager& vm, int count);
    int  selectNextMove(void) const;

private:
    Node*  selectMostVisitedChild(void) const;
    void   setChildAndRemoveTabu(int child_customer);
    double computeUcb(int parent_count);

    int  customer_;
    int  count_;
    int  child_size_;
    int  value_;
    Node *child_;
    bool *tabu_;
};

#endif /* VRPSOLVER_CPP_NODE_H */
