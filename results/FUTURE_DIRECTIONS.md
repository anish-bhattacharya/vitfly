# Future Directions: Beyond Behavior Cloning

A brainstorm based on literature review (May 2026).

## 1. RL Fine-Tuning on Top of BC (Bootstrapping RL with IL)

**Reference**: Bootstrapping RL with Imitation for Vision-Based Agile Flight (CoRL 2024, Xing et al.)

**Concept**: Three-stage pipeline proven for quadrotor racing:
1. Train teacher policy with **privileged information** (state-based) via RL
2. Distill to student policy via **Behavior Cloning** (vision-based)
3. **Adaptive RL fine-tuning** on the student policy

**Why it applies here**: We already have a working BC policy (Branch B). Stage 3 (RL fine-tuning) is the incremental step. The RL reward can be `collision_avoidance + path_progress - time_penalty`.

**Effort**: High (need RL training loop + simulator integration)
**Expected gain**: Robustness improvement, handles covariate shift

## 2. DAgger (Dataset Aggregation)

**Reference**: DAgger for Autonomous Driving (Ross et al., A* DAgger 2021)

**Concept**: Iteratively collect data from the learned policy, get expert labels, and retrain:
1. Rollout current policy → collects states the policy actually visits
2. Expert relabels those states with correct actions
3. Aggregate into training set → retrain

**Why it applies**: Our expert (Flightmare simulator with privileged planner) can be queried on-policy. The A* planner variant shows pseudo-experts work.

**Effort**: Medium (need DAgger loop + expert query mechanism)
**Expected gain**: Addresses covariate shift, our primary failure mode

## 3. Curriculum Learning: Easy-to-Hard Terrain Generation

**Reference**: Eurekaverse (LLM-based curriculum, arXiv 2024), GACL (ICRA 2025)

**Concept**: Use LLM/algorithm to generate terrain of increasing difficulty:
1. Start with simple obstacles (single sphere in open space)
2. Progressively add: narrow corridors → U-shaped walls → dense forests → dynamic obstacles
3. Adapt difficulty based on policy performance

**Why it applies**: Our current data has fixed difficulty distribution. A curriculum lets the policy master simple cases before tackling hard ones.

**Effort**: High (need terrain generator + difficulty scoring)
**Expected gain**: Smoother training, better generalization

## 4. Asymmetric Expert: Reduce Student-Expert Gap

**Reference**: LEAD: Minimizing Learner-Expert Asymmetry (NeurIPS 2024)

**Concept**: Three types of expert-student mismatch:
- **Visibility**: Expert sees privileged info (e.g., obstacle centers) → student doesn't
- **Uncertainty**: Expert on noise-free state → student on noisy vision
- **Intent**: Expert's navigation goal underspecified

**Why it applies**: Our expert uses full privileged state. Our student sees only depth images. Reducing this gap (e.g., adding noise to expert during demonstration) produces more learnable demonstrations.

**Effort**: Low-Medium (modify expert data generation)
**Expected gain**: Better BC performance without changing model architecture

## 5. Data Augmentation via Counterfactual/APC

**Reference**: Augmented Policy Cloning (NeurIPS 2022), CF-Driver (2024)

**Concept**: Generate additional training data by:
- **APC**: Perturb the state slightly, query expert for corrected action → teaches feedback responses
- **CF**: Generate edge cases near decision boundaries (e.g., obstacle just barely on left vs right)
- **Dynamic agents**: Extract trajectories from nearby dynamic objects as additional expert data

**Why it applies**: Our 42K-image dataset is small by modern standards. These methods can generate 5-10x more data without new simulation runs.

**Effort**: Medium (need expert query loop)
**Expected gain**: 2-5x data efficiency improvement

## 6. Multi-Agent: Dynamic Obstacle Generator for Easy-to-Hard

**Reference**: Curriculum RL from Avoiding Collisions (IEEE RAL 2023)

**Concept**: A generative agent that spawns obstacles with adaptive difficulty:
- Easy mode: static spheres, large gaps
- Medium mode: moving obstacles, narrow corridors
- Hard mode: adversarial obstacle placement, occluded paths
- Dynamic: adapt difficulty based on policy success rate

**Why it applies**: Directly addresses your idea of "a generative agent that creates dynamically changing terrain for easy2hard training."

**Effort**: Very High (need multi-agent training + environment generation)
**Expected gain**: Most comprehensive training regime

## Implementation Priority

| # | Approach | Effort | Expected Gain | Dependencies |
|---|----------|--------|---------------|--------------|
| 1 | **DAgger** | Medium | High | Expert query, rollout pipeline |
| 2 | **Asymmetric expert** | Low | Medium | Expert config change |
| 3 | **APC augmentation** | Medium | High | Expert query loop |
| 4 | **RL fine-tuning** | High | Very High | RL env + reward design |
| 5 | **Curriculum terrain** | Very High | High | Terrain generator |
| 6 | **Multi-agent terrain** | Very High | Very High | Multi-agent env |

**Recommended first step**: Start with #2 (asymmetric expert, low effort) + #1 (DAgger, high impact). The infrastructure for #1 also enables #3 and #4.

## Key Literature

1. Xing et al., "Bootstrapping RL with Imitation for Vision-Based Agile Flight" (CoRL 2024)
2. Codevilla et al., "Exploring the Limitations of Behavior Cloning" (ICCV 2019)
3. LEAD: "Minimizing Learner-Expert Asymmetry" (NeurIPS 2024)
4. APC: "Data Augmentation for Efficient Learning from Parametric Experts" (NeurIPS 2022)
5. Eurekaverse: "Environment Curriculum Generation via LLMs" (2024)
6. GACL: "Grounded Adaptive Curriculum Learning" (2025)
