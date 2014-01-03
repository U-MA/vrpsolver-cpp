#ifndef VRPSOLVER_CPP_NODE_H
#define VRPSOLVER_CPP_NODE_H

class Node
{
private:
    int  count_;
    int  childSize_;
    Node *child;

public:
    Node(void);
    ~Node(void);

    int count(void) const;
    int childSize(void) const;
    void expand(int childSize);
    Node *select(void);
    bool isLeaf(void) const;
};

#endif /* VRPSOLVER_CPP_NODE_H */
