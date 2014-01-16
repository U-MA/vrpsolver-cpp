#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
#include "vrp_types.h"
}

#include "Node.h"
#include "VehicleManager.h"
#include "VrpSimulation.h"


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

/* childに値をセットする以上のことをしているので
 * 関数名を変えた方がいいかも */
void Node::setChild(int child_customer)
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

    /* 次の車体が存在 */
    if (vm.nextVehicleRemain(vrp))
        setChild(VehicleManager::kChange);

    /* 各顧客が訪問可能か調べる */
    for (int i=1; i < vrp->vertnum; i++)
        if (!vm.isVisit(i) && vm.canVisit(vrp, i))
            setChild(i);
}

double Node::computeUcb(int parent_count)
{
    double ucb = 1e6 + (rand() % (int)1e6);
    if (count_ != 0)
        ucb = - value_ / count_ + 1.0 * sqrt(log((double)parent_count+1)) / count_;

    return ucb;
}

Node *Node::select(void)
{
    /* MISSっていう値はわかりづらい.INFとかのほうがわかりやすいのでは? */
    double max_ucb   = - VrpSimulation::kInfinity;
    Node   *selected = NULL;

    for (int i=0; i < child_size_; i++)
    {
        /* tabu_に含まれているものは選択しない */
        if (tabu_[child_[i].customer()]) continue;

        double ucb = child_[i].computeUcb(count_);
        //fprintf(stderr, "child[%d].computeUcb(%d) is %lg and child[%d].count() is %d\n", i, count_, ucb, i, child[i].count());
        if (ucb > max_ucb)
        {
            max_ucb  = ucb;
            selected = &child_[i];
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


void Node::search(const vrp_problem *vrp, const VehicleManager& vm, int count)
{
    /* 引数として渡されるvmは変更しない
     * そのため変更させるための変数を作成 */
    VehicleManager vm_copy = vm.copy();

    Node *visited[300];
    int  visited_size = 0;

    Node *node   = this;
    Node *parent = NULL;

    /* 木の根は訪問済み */
    visited[visited_size++] = this;

    //fprintf(stderr, "\nMONTE CARLO TREE ROOT address %p IS ", node);

    /* SELECTION */
    while (!node->isLeaf())
    {
        //fprintf(stderr, "NODE\n");
        parent = node;
        node   = node->select();
        //fprintf(stderr, "\tNODE address %p (HAVE CUSTOMER %d) IS ", node, node->customer());
        if (!node->isLeaf() && node->isTabu(vrp))
        {
            //fprintf(stderr, "TABU\n");
            parent->addTabu(node->customer());
            return ; /* 探索を破棄 */
        }
        visited[visited_size++] = node;
        vm_copy.move(vrp, node->customer());
    }

    //fprintf(stderr, "LEAF\n");

    /* 後の操作で現在のVehiceManaagerの状態を使う必要が
     * あるかもしれないので、ここで記憶しておく
     * 囲碁将棋の「待った」から来ている */
    VehicleManager matta = vm_copy.copy();

    /* nodeが全探索木の葉でなければexpandする*/
    if (!vm_copy.isFinish(vrp))
    {
        /* EXPANSION */
        //fprintf(stderr, "\tEXPAND\n");
        node->expand(vrp, vm_copy);
        parent = node;
        node   = node->select();
        //fprintf(stderr, "\t\tSELECTED NODE address %p HAVE CUSTOMER %d\n", node, node->customer());
        vm_copy.move(vrp, node->customer());
    }

    /* SIMULATION */
    int cost = VrpSimulation::kInfinity;
    while ((cost = VrpSimulation::sequentialRandomSimulation(vrp, vm_copy, count)) == VrpSimulation::kInfinity)
    {
        //fprintf(stderr, "\t\t[SIMULATION RESULT] %d\n", cost);
        //fprintf(stderr, "\t\t\tSO, NODE address %p ADD CUSTOMER %d TO TABU\n", parent, node->customer());
        parent->addTabu(node->customer());
        vm_copy = matta.copy(); /* VehicleManagerを直前の状態に移す */
        if (parent->isTabu(vrp))
        {
            //fprintf(stderr, "\t\t\tPARENT NODE address %p IS TABU\n", parent);
            return ; /* 探索を破棄 */
        }
        node = parent->select();
        vm_copy.move(vrp, node->customer());
    }
    //fprintf(stderr, "\t\t[SIMULATION RESULT] %d\n", cost);
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
