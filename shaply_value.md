## Shapley Value
In cooperative game theory, the Shapley value is a method (solution concept) for fairly distributing the total gains or costs among a group of players who have collaborated.
![alt text](images/shapley_value1.png)


Example: 在一个联盟游戏（前面描述的场景）中，我们有一组 N 个玩家。我们还有一个函数 v，它给出了这些参与者的任何子集的值，也就是说，S 是 N 的子集，然后 v（S）给出了该子集的值。因此，对于一个联合博弈（N，v），我们可以使用这个方程来计算玩家 i 的贡献，即 Shapley 值。

![alt text](images/shapley_value2.png)

## Reference
1. "Shapley Value Based Multi-Agent Reinforcement Learning: Theory, Method and Its Application to Energy Network" [[PHD Thesis]](https://arxiv.org/abs/2402.15324)
2. "Shapley Library" [[Code]](https://shap.readthedocs.io/en/latest/index.html) [[Shapley Value + PPO]](https://github.com/MaximeSzymanski/PPO)
3. "Adaptive Value Decomposition with Greedy Marginal Contribution Computation for Cooperative Multi-Agent Reinforcement Learning" [[Paper]](https://arxiv.org/abs/2302.06872)
3. https://zhuanlan.zhihu.com/p/91834300
4. https://en.wikipedia.org/wiki/Shapley_value

