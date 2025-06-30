## Heterogeneous Multi-robot System Cooperation

1. "Heterogeneous Multi-Robot Reinforcement Learning". [[Paper]](https://arxiv.org/pdf/2301.07137) [[Code]](https://github.com/proroklab/HetGPPO) [[BenchMARL]](https://matteobettini.com/publication/benchmarl/) [[TorchRL]](https://github.com/pytorch/rl/tree/main/sota-implementations/multiagent) [[VMAS]](https://github.com/proroklab/VectorizedMultiAgentSimulator) 
    * HetGPPO learns individual agnet policies: 1) Use neighbourhood communication to overcome partial observability. 2) Allows decentralized training of GNNs.
    * This study found that homogeneous agents are able to infer behavioral roles through observations, emulating heterogeneous behavior, which is call **behavioral typing**.
    <img src="images/taxonomy_HMRs.png" alt="taxonomy_HMRs" width="600"/>
    <img src="images/Behavioral_typing.png" alt="taxonomy_HMRs" width="600"/>
    <img src="images/HetGPPO_results.png" alt="taxonomy_HMRs" width="600"/>    
    <img src="images/ecosystem_MARL.png" alt="taxonomy_HMRs" width="600"/>
2. "Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning" [[Paper]](https://arxiv.org/abs/2405.15054) [[Code]]()
3. "When Is Diversity Rewarded in Cooperative Multi-Agent Learning?" [[Paper]](https://arxiv.org/pdf/2506.09434)
4. "Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers". [[Paper]](https://arxiv.org/abs/2409.20537) [[Code]](https://github.com/liruiw/HPT) [[Gym-Aloha]](https://github.com/huggingface/gym-aloha)
5. "Transformer-based Multi-Agent Reinforcement Learning for Generalization of Heterogeneous Multi-Robot Cooperation" [[Paper]](https://ieeexplore.ieee.org/document/10802580) [[Platform]](https://shubhlohiya.github.io/MARBLER/)
6. "MARBLER: An Open Platform for Standardized Evaluation of Multi-Robot Reinforcement Learning Algorithms" [[Paper]](https://arxiv.org/abs/2307.03891) [[Code]](https://github.com/GT-STAR-Lab/MARBLER)
7. "Learning Heterogeneous Agent Collaboration in Decentralized Multi-Agent Systems via Intrinsic Motivation" [[Paper]](https://arxiv.org/pdf/2408.06503) [[Code]](https://github.com/jahirsadik/CoHet-Implementation)
8. "Heuristics-Assisted Experience Replay Strategy for Cooperative Multi-Agent Reinforcement Learning" [[Paper]](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2798.pdf)
9. "Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities". [[Paper]](https://openreview.net/forum?id=N3VbFUpwaa) [[Code]](https://github.com/GT-STAR-Lab/cap-comm)
10. "Heterogeneous Multi-Agent Reinforcement Learning for Zero-Shot Scalable Collaboration" [[Paper]](https://arxiv.org/abs/2404.03869)
11. "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games". [[Tutorial]](https://docs.pytorch.org/rl/0.4/tutorials/multiagent_ppo.html#) [[Paper]](https://arxiv.org/abs/2103.01955) [[Code]](https://github.com/marlbenchmark/on-policy)[[MARL-Algorithms]](https://github.com/pytorch/rl/tree/main/sota-implementations/multiagent)[[MPEs)]](https://github.com/openai/multiagent-particle-envs) [[GRF]](https://github.com/google-research/football) [[StarCraftII v2]](https://github.com/oxwhirl/smacv2) 
    * The main difference between Independent Proximal Policy Optimization (IPPO) and Multi-Agent Proximal Policy Optimization (MAPPO) lies in how they handle the value function (critic) during training. IPPO uses a decentralized critic, where each agent has its own independent critic network. MAPPO, on the other hand, employs a centralized critic that can access information about the global state or the observations of all agents, even when training is decentralized.
    * IPPO is a straightforward multi-agent extension of the single-agent PPO, where each agent acts independently.
    * MAPPO builds upon PPO by introducing a centralized critic to leverage global information for better learning, particularly in cooperative settings. 
12. "Selectively Sharing Experiences Improves Multi-Agent Reinforcement Learning" [[Paper]](https://openreview.net/forum?id=DpuphOgJqh&noteId=UhjBgbqvOt) [[Code]]()
13. "Fully Decentralized Cooperative Multi-Agent Reinforcement Learning: A Survey" [[Paper]](https://arxiv.org/pdf/2401.04934) [[Code]]()
14. "Decentralized Multi-Robot Navigation for Autonomous Surface Vehicles with Distributional Reinforcement Learning" [[Paper]](https://arxiv.org/abs/2402.11799) [[Code]](https://github.com/RobustFieldAutonomyLab/Multi_Robot_Distributional_RL_Navigation)
15. "Measuring Policy Distance for Multi-Agent Reinforcement Learning" [[Paper]](https://arxiv.org/pdf/2401.11257) [[Code]](https://github.com/Harry67Hu/MADPS)
16. "Heterogeneity in Multi-agent Systems" [[PhD Thesis]](https://research-information.bris.ac.uk/ws/portalfiles/portal/405101389/Heterogeneity_in_Multi_Agent_Systems.pdf)
17. "Multi-robot Exploration with RL" [[Code]](https://github.com/i1Cps/multi-robot-exploration-rl)
18. "PPO with transformers" [[Code]](https://github.com/datvodinh/ppo-transformer/tree/main)
119. "Cooperative and Asynchronous Transformer-based Mission Planning for Heterogeneous Teams of Mobile Robots" [[Paper]](https://arxiv.org/abs/2410.06372) [[Code]](https://arxiv.org/abs/2410.06372)