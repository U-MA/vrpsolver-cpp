#include <math.h>

extern "C"
{
#include "vrp_types.h"
}

#include "Node.h"
#include "VehicleManager.h"
#include "VrpSimulation.h"

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

double Node::computeUcb(int parentCount)
{
    double ucb = 1e6;
    if (count_ != 0)
        ucb = value_ / count_ + 1.0 * sqrt(log((double)parentCount+1)) / count_;

    return ucb;
}

Node *Node::select(void)
{
    Node *selected = NULL;
    double maxUcb = -1.0;
    for (int i=0; i < childSize_; i++)
    {
        double ucb = child[i].computeUcb(count_);
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

void Node::search(const vrp_problem *vrp, const VehicleManager& vm, const Vehicle& v)
{
    printf("in search\n");
    /* 引数として渡されるvm, vは変更しない
     * そのため変更させるための変数を作成 */
    VehicleManager vm_copy = vm.copy();
    Vehicle         v_copy =  v.copy();

    Node *visited[300];
    int  visitedSize = 0;

    Node *node = this;

    /* nodeは訪問済み */
    visited[visitedSize++] = this;

    /* SELECTION */
    while (!node->isLeaf())
    {
        node = node->select();
        visited[visitedSize++] = node;
        vm_copy.move(vrp, node->customer());
    }

    /* EXPANSION */
    /* 顧客の数+1の子を作成. +1は車体の変更 */
    node->expand(vrp->vertnum);
    Node *newNode = node->select();
    visited[visitedSize++] = newNode;
    vm_copy.move(vrp, newNode->customer());

    /* SIMULATION */
    int cost = VrpSimulation::sequentialRandomSimulation(vrp, vm_copy, v_copy);

    /* BACKPROPAGATION */
    for (int i=0; i < visitedSize; i++)
        visited[i]->update(cost);
}

void Node::search(const vrp_problem *vrp, const VehicleManager& vm)
{
}

int Node::next(void) const
{
    int maxCount = -1;
    int move     = -1;
    for (int i=0; i < childSize_; i++)
    {
        int count = child[i].count();
        if (count > maxCount)
        {
            maxCount = count;
            move     = child[i].customer();
        }
    }

    return move;
}
