#ifndef VRPSOLVER_CPP_NODE_H
#define VRPSOLVER_CPP_NODE_H

class Node
{
private:
    int  customer_;
    int  count_;
    int  childSize_;
    int  value_;
    Node *child;

public:
    Node(void);
    ~Node(void);

    int customer(void) const;
    int count(void) const;
    int childSize(void) const;
    int value(void) const;
    void expand(int childSize);
    Node *select(void);
    void update(int value);
    bool isLeaf(void) const;
};

#endif /* VRPSOLVER_CPP_NODE_H */
