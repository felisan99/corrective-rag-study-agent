# Binary Search Trees

## Definition

A binary search tree (BST) is a binary tree data structure where each node
has at most two children, referred to as the left child and the right
child. For every node `n` in the tree, all keys in the left subtree of `n`
are smaller than `n`'s key, and all keys in the right subtree of `n` are
greater than `n`'s key. This ordering property is what makes efficient
search possible.

## Core operations

- **Search**: Start at the root and compare the target key to the current
  node's key. If it matches, return the node. If the target is smaller, move
  to the left child; if larger, move to the right child. Repeat until the
  key is found or a null child is reached (meaning the key is not present).
- **Insertion**: Follow the same comparison path used in search until an
  empty spot is found, then attach the new node there as a leaf.
- **Deletion**: Three cases: deleting a leaf (just remove it), deleting a
  node with one child (replace the node with its child), and deleting a
  node with two children (replace the node's value with its in-order
  successor -- the smallest value in its right subtree -- then delete that
  successor node).

## Time complexity

For a BST with `n` nodes:

- If the tree is **balanced** (roughly the same number of nodes in the left
  and right subtrees at every level), search, insertion, and deletion all
  run in **O(log n)** time, because each comparison eliminates about half
  of the remaining nodes.
- If the tree is **unbalanced** -- for example, if keys are inserted in
  already-sorted order, producing a tree that degenerates into a linked
  list -- these operations run in **O(n)** time in the worst case.

## Balancing

Because a plain BST offers no guarantee of balance, self-balancing variants
were developed to guarantee O(log n) operations regardless of insertion
order:

- **AVL trees** rebalance after every insertion/deletion by tracking the
  height difference (balance factor) between each node's subtrees and
  performing rotations when that difference exceeds 1.
- **Red-Black trees** use a coloring scheme (each node is red or black)
  with a small set of invariants that bound the tree's height to
  approximately `2 * log(n)`, needing fewer rotations than AVL trees on
  average, which is why they're the underlying structure for many standard
  library ordered maps/sets (e.g. C++'s `std::map`, Java's `TreeMap`).

## In-order traversal

Visiting a BST's nodes in-order (left subtree, node, right subtree)
produces the keys in sorted ascending order. This is a direct consequence
of the BST ordering property and is commonly used to validate that a tree
satisfies the BST invariant.
