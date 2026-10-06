# Day 4

## 🎯 Focus
Gradient descent in practice, and learning how to safely undo things in Git.

## 📚 Learned
- Gradient descent improves a model by repeatedly nudging its parameters in the direction that reduces the loss.
- The learning rate controls the step size: too small and training crawls, too large and the loss bounces around or blows up.
- `git reset` moves the branch pointer back (rewriting history), while `git revert` adds a new commit that undoes an old one without rewriting anything.
- `git stash` temporarily shelves uncommitted changes so I can switch tasks and bring them back later.
- Basic rebase replays my commits on top of another branch to keep history linear.
- Tags mark specific commits (like releases) with a readable name.
- Good commit messages are short, in the imperative mood, and say what changed and why.

## 💻 Built
- Ran gradient descent with different learning rates and recorded how the loss behaved for each.
- Added basic tests to the project.

## 🧩 DSA
- [LeetCode 34 – Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

## ✅ Done when
- [ ] Gradient descent lab completed
- [ ] Practised reset, revert, stash, rebase basics and tags
- [ ] Learning-rate experiment results recorded
- [ ] Basic tests added