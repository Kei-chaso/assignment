**Minimization of Surprise (Objective Function)**
Transformers are trained to minimize <abbr title='A mathematical measurement quantifying the difference between two probability distributions, frequently used to evaluate the accuracy of predictive models by penalizing incorrect classifications. /ˌkɹɔs ˈɛn.tʰɹə.pi lɔs/'>cross-entropy loss</abbr> (predicting the next token), which represents the negative log-likelihood of the data. The Free Energy Principle (FEP) relies on minimizing variational free energy, an upper bound on surprise (the negative log-probability of sensory inputs).[^annotation_93409a0fb903] Both frameworks fundamentally optimize their internal models by minimizing prediction error.

**Hierarchical Generative Modeling**
Transformers utilize deep, stacked layers to construct increasingly abstract, hierarchical representations of sequential data. FEP postulates that intelligent systems operate as hierarchical generative models, where higher levels generate top-down predictions to explain lower-level states, extracting abstract causes from raw data.

**Attention as Precision Weighting**
In FEP, attention is conceptualized as precision optimization—weighting sensory prediction errors based on their reliability (inverse variance). The Transformer’s self-attention mechanism performs an analogous function by dynamically assigning weights to context tokens, determining which pieces of information are most relevant and reliable for updating the current state.

**Latent State Inference (Bayesian Updating)**
A Transformer iteratively updates the latent representation (hidden states) of each token by integrating information from its surrounding context during the forward pass. This parallels Bayesian belief updating in FEP, where an agent continuously updates its internal latent states based on new sensory evidence to align its generative model with the environment.

**Predictive Processing**
Autoregressive Transformers are inherently predictive engines, continuously calculating the probability distribution of the next state based on prior context. This directly mirrors the core FEP concept of the brain as a predictive coding system that constantly anticipates future sensory inputs to minimize future surprise and maintain equilibrium.

[^annotation_93409a0fb903]: **Original Text:**
    The Free Energy Principle (FEP) relies on minimizing variational free energy, an upper bound on surprise (the negative log-probability of sensory inputs).

    **Translation:**
    自由エネルギー原理（FEP）は、サプライズ（感覚入力の負の対数確率）の上限である変分自由エネルギーを最小化することに基づいている。

    **Explanation:**
    The Free Energy Principle (FEP) is a prominent theoretical framework in neuroscience and cognitive science. It proposes that all biological systems, such as the human brain, maintain their existence by constantly striving to remain in predictable, safe states. To achieve this, they must avoid "surprise."

    In this context, "surprise" (also called surprisal) is not a psychological emotion, but a precise mathematical concept from information theory. It is defined as the "negative log-probability of sensory inputs." In simpler terms, it measures how mathematically unlikely a piece of sensory information is, given the organism's model of the world.
    - If a sensory input is highly probable (e.g., feeling gravity pull you downward), the negative log-probability is a small number, meaning "surprise" is low.
    - If a sensory input is highly improbable (e.g., suddenly floating weightless in your living room), the negative log-probability is a large number, meaning "surprise" is high. 
    Biologically, organisms must minimize surprise because prolonged exposure to highly unexpected states usually means danger or death (for example, a fish suddenly experiencing the sensory input of dry air).

    However, there is a fundamental computational problem: the brain cannot calculate true "surprise" directly. To compute the exact objective probability of a sensory input, the brain would need to know the true, hidden state of the entire external universe, which is impossible. The brain is locked inside a skull and only has access to its own internal models and the limited sensory data it receives.

    To solve this, the brain uses a mathematical proxy called "variational free energy." Variational free energy is a quantity that the brain *can* actually compute because it relies only on things the brain has access to: its internal predictions and the incoming sensory signals. 

    Crucially, variational free energy acts as an "upper bound" on surprise. In mathematics, if value A (free energy) is an upper bound on value B (surprise), A is always greater than or equal to B. Therefore, if the brain actively pushes down (minimizes) the variational free energy, it mathematically guarantees that the true, unknowable surprise is also being pushed down. 

    The brain minimizes this free energy in two primary ways:
    1. Perceptual Inference / Learning: Updating its internal models so that its predictions better match reality, thus reducing prediction errors.
    2. Active Inference / Action: Moving the body to change the environment or the sensory inputs so that they match what the brain already predicts (e.g., if you predict you are warm, but sense cold, you minimize free energy by putting on a jacket to make the sensory input match your prediction).

