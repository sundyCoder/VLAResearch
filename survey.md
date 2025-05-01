## 1. Robotic Learning Directions
- Vision-Language-Model (VLM): VLM act as an ageent that can genrate task execution function code. e.g, ReKep, Voxposer
- Behaviror cloning(BC): learning from demonstration data and generate action sequences. e.g., Action chunking, Policy diffusion, Flow matching
- Vision-language-action (VLA): genration action sequence end-to-end. e.g., Issac GR001, Pi-0, OpenVLA.
- VLM + VLA: slow think (VLM) for planning and fast think for action generation (VLA). e.g., [figureAI](https://www.figure.ai/)
![alt text](images/figure-AI.png)

    System 2 (S2): An onboard internet-pretrained VLM operating at 7-9 Hz for scene understanding and language comprehension, enabling broad generalization across objects and contexts.

    System 1 (S1): A fast reactive visuomotor policy that translates the latent semantic representations produced by S2 into precise continuous robot actions at 200 Hz.


## 2. Vision-Language-Model
Input the image observation of robots to large-vision model and the human prompt to vision luange model, finally genration the code that contains the tasks execution functions.

Please see the below typical research article named ReKep:
![alt text](images/ReKep.png)

### Reference
[1]. [Rekep spatio-temporal reasoning of relational keypoint constraints for robotic manipulation](https://arxiv.org/abs/2409.01652)

## 2. Behavior clone
### 2.1 Action Chunking
Action chunking is an open-loop control where at every control time step, a policy outputs a chunk (sequence) of actions into the future given the current observation. Usually the action sequence will be fully or partially executed before the next control time step.

For example, in robotic manipulation, the action "set the table" can be chunked into smaller steps such as "pick up the plate," "place the plate on the table," and "adjust the plate's position."

This is in contrast to the typical closed-loop control, where the policy observes the current observation, outputs only one action, observes the next observation, and so on.
![alt text](images/closed_loop-vs-action_chunking.png)

### 2.2 Diffusion Policy ([Chi et al. 2023](https://www.bilibili.com/video/BV1ZaeAe7EMu/?spm_id_from=333.337.search-card.all.click&vd_source=5acd2d369a16a68747ef0223c2f4a7a4))
![alt text](images/diffusion_policy.png)

![alt text](images/action_modality.png)

#### 2.2.1 Action Space Scalability: easily afford action sequence prediction
![alt text](images/action_space_scalability.png)

## 3. VLA Research Questions
Survey paper: [Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models](https://robovlms.github.io/)

![image](images/research_questions.png)

## 3.1 Vision-language-action(VLA)
### 3.1.1 VLA Formuation
![image](images/vla_formulation.png)

### 3.1.2 Backbone architecture

Flamingo and OFA: 1) encoder-decoder -- an encoder that is typically responsible for extracting features from inputs using input embedding modules as discussed above, and a decoder that generates the output (e.g., text or multi-modal predictions) auto-regressive.

LLaVA, GPT-4V: 2)Decoder-only -- Decoder-only architectures, in contrast, rely on a unified transformer framework where both the input modalities (vision and text) and the output sequences are processed in the same auto-regressive decoder. 

### 3.1.3 Action Prediction
<!-- ![image](images/action_prediction.png) -->
<img src="images/action_prediction.png" alt="alt text" width="500"/>

![alt text](images/actions.png)

### 3.1.4 Flow Matching ([Physical Intelligence 2024](https://www.physicalintelligence.company/download/pi0.pdf).)

Flow Matching is a training approach inspired by diffusion models, designed to align the flow of data across different modalities (e.g., vision, language, and action). It enables Vision-Language-Action (VLA) models to handle both discrete (e.g., language tokens) and continuous (e.g., action trajectories) data within a unified framework.

This method improves the model's ability to predict action sequences by ensuring that the flow of information across modalities is consistent. For example, in robotic control, Flow Matching helps align visual observations, language instructions, and action trajectories to generate precise and coherent action sequences.

Inspired by [Transfusion](https://arxiv.org/abs/2408.11039), Flow Matching combines the language modeling loss function (next token prediction) with diffusion techniques to train a single transformer over mixed-modality sequences.

## 4. Open source datasets
![image](images/exp_env.png)

## 5. Reference
[1]. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705) \
[2]. [Action chunking vs closed-loop control](https://www.haonanyu.blog/post/action_chunking/) \
[3]. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) [[Source Code]](https://github.com/real-stanford/diffusion_policy) [[Video]](https://www.bilibili.com/video/BV1ZaeAe7EMu/?spm_id_from=333.337.search-card.all.click&vd_source=5acd2d369a16a68747ef0223c2f4a7a4 ) \
[4]. [π0: A Vision-Language-Action Flow Model for
General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf) \
[5]. [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039) \
[6]. [Issac Sim](https://www.nvidia.cn/training/learning-path/robotics/)

