#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
#include "vrp_types.h"
}

#include "node.h"
#include "vehicle_manager.h"
#include "simulator.h"


Node::~Node(void)
{
    delete[] child_;
    delete[] tabu_;
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
    return child_size_;
}

int Node::value(void) const
{
    return value_;
}

bool Node::tabu(int customer) const
{
    return tabu_[customer];
}

void Node::addTabu(int customer)
{
    tabu_[customer] = true;
}

void Node::setChildAndRemoveTabu(int child_customer)
{
    child_[child_size_++].customer_ = child_customer;
    tabu_[child_customer]           = false; /* tabu_リストから外す */
}

void Node::expand(const vrp_problem *vrp, VehicleManager& vm)
{
    child_ = new Node[vrp->vertnum];
    tabu_  = new bool[vrp->vertnum];
    for (int i=0; i < vrp->vertnum; i++)
        addTabu(i);

    if (vm.nextVehicleRemain(vrp))
        setChildAndRemoveTabu(VehicleManager::kChange);

    for (int i=1; i < vrp->vertnum; i++)
        if (!vm.isVisit(i) && vm.isInCapacityConstraint(vrp, i))
            setChildAndRemoveTabu(i);
}

double Node::computeUcb(int parent_count)
{
    double ucb = 1e6 + (rand() % (int)1e6);
    if (count_ != 0)
        ucb = - value_ / count_ + 1.0 * sqrt(log((double)parent_count+1)) / count_;

    return ucb;
}

Node *Node::selectMaxUcbChild(void)
{
    double max_ucb   = - 1e6;
    Node   *selected = NULL;

    for (int i=0; i < child_size_; i++)
    {
        Node *child        = &child_[i];
        bool child_is_tabu = tabu_[child->customer()];

        if (child_is_tabu) continue;

        double ucb = child->computeUcb(count_);
        if (ucb > max_ucb)
        {
            max_ucb  = ucb;
            selected = child;
        }
    }

    return selected;
}

bool Node::isLeaf(void) const
{
    return (child_size_ == 0);
}

bool Node::isTabu(const vrp_problem *vrp) const
{
    for (int i=0; i < vrp->vertnum; i++)
        if (!tabu_[i])
            return false;

    return true;
}

void Node::update(int value)
{
    count_++;
    value_ += value;
}


void Node::build(const vrp_problem *vrp, const VehicleManager& vm, int count)
{
    /* 引数として渡されるvmは変更しない
     * そのため変更させるための変数を作成 */
    VehicleManager vm_copy = vm;

    Node *visited[300];
    int  visited_size = 0;

    Node *parent = NULL;
    Node *node   = this;

    /* 木の根は訪問済み */
    visited[visited_size++] = this;

    /* SELECTION */
    while (!node->isLeaf())
    {
        if (node->isTabu(vrp))
        {
            parent->addTabu(node->customer());
            return ; /* 探索を破棄 */
        }
        parent = node;
        node   = node->selectMaxUcbChild();
        visited[visited_size++] = node;
        vm_copy.move(vrp, node->customer());
    }

    VehicleManager parent_vm = vm_copy;

    /* nodeが全探索木の葉でなければexpandする*/
    if (vm_copy.canMove(vrp))
    {
        /* EXPANSION */
        node->expand(vrp, vm_copy);
        parent = node;
        node   = node->selectMaxUcbChild();
        vm_copy.move(vrp, node->customer());
    }

    /* SIMULATION */
    int cost = 1e6;
    Simulator simulator;
    while ((cost = simulator.sequentialRandomSimulation(vrp, vm_copy, count)) == 1e6)
    {
        parent->addTabu(node->customer());
        vm_copy = parent_vm; /* VehicleManagerを直前の状態に移す */
        if (parent->isTabu(vrp))
            return ; /* 探索を破棄 */

        node = parent->selectMaxUcbChild();
        vm_copy.move(vrp, node->customer());
    }
    visited[visited_size++] = node;

    /* BACKPROPAGATION */
    for (int i=0; i < visited_size; i++)
        visited[i]->update(cost);
}

Node *Node::selectMostVisitedChild(void) const
{
    Node *selected = NULL;
    int maxCount   = -1;
    for (int i=0; i < child_size_; i++)
    {
        int count = child_[i].count();
        if (count > maxCount)
        {
            maxCount = count;
            selected = &child_[i];
        }
    }
    return selected;
}

int Node::selectNextMove(void) const
{
    Node *selected = selectMostVisitedChild();
    return selected->customer();
}
