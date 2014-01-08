#include <math.h>
#include <stdlib.h>

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

void Node::expand(const vrp_problem *vrp, VehicleManager& vm)
{
    int childSize = 0;
    child = new Node[vrp->vertnum];

    /* 次の車体が存在 */
    if (vm.size() < vrp->numroutes)
        child[childSize++].customer_ = VehicleManager::CHANGE;

    /* 各顧客が訪問可能か調べる */
    for (int i=1; i < vrp->vertnum; i++)
    {
        if (!vm.isVisit(i) && vm.canVisit(vrp, i))
            child[childSize++].customer_ = i;
    }
    childSize_ = childSize;
}

double Node::computeUcb(int parentCount)
{
    double ucb = 1e6 + (rand() % (int)1e6);
    if (count_ != 0)
        ucb = - value_ / count_ + 1.0 * sqrt(log((double)parentCount+1)) / count_;

    return ucb;
}

Node *Node::select(void)
{
    Node *selected = NULL;
    double maxUcb = -INF;
    for (int i=0; i < childSize_; i++)
    {
        double ucb = child[i].computeUcb(count_);
        //printf("child[%d].computeUcb(%d) is %lg and child[%d].count() is %d\n", i, count_, ucb, i, child[i].count());
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

void Node::search(const vrp_problem *vrp, const VehicleManager& vm)
{
    /* 引数として渡されるvmは変更しない
     * そのため変更させるための変数を作成 */
    VehicleManager vm_copy = vm.copy();

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
        //printf("node->customer() is %d\n", node->customer());
        vm_copy.move(vrp, node->customer());
    }

    /* nodeが全探索木の葉でなければexpandする*/
    if (!vm_copy.isFinish(vrp))
    {
        /* EXPANSION */
        node->expand(vrp, vm_copy);
        Node *newNode = node->select();
        visited[visitedSize++] = newNode;
        //printf("newNode->customer() is %d\n", newNode->customer());
        vm_copy.move(vrp, newNode->customer());
    }

    /* SIMULATION */
    int cost = VrpSimulation::sequentialRandomSimulation(vrp, vm_copy);

    /* BACKPROPAGATION */
    for (int i=0; i < visitedSize; i++)
        visited[i]->update(cost);
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
