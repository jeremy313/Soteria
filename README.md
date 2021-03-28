# Provable-Defense-against-Privacy-Leakage-in-Federated-Learning-from-Representation-Perspective
Official implementation of "Provable Defense against Privacy Leakage in Federated Learning from Representation Perspective"

The paper can be found at https://arxiv.org/pdf/2012.06043.pdf

# Abstract

Federated learning (FL) is a popular distributed learning framework that can reduce privacy risks by not explicitly sharing private data. %It can reduce privacy risks. 
However, recent works have demonstrated that sharing model updates makes FL vulnerable to inference attack. In this work, we show our key observation that the data representation leakage from gradients is the essential cause of privacy leakage in FL. We also provide an analysis of this observation to explain how the data presentation is leaked. 
Based on this observation, we propose a defense against model inversion attack in FL. The key idea of our defense is learning to perturb data representation such that the quality of the reconstructed data is severely degraded, while FL performance is maintained. In addition, we derive certified robustness guarantee to FL and convergence guarantee to FedAvg after applying our defense. 
To evaluate our defense, we conduct experiments on MNIST and CIFAR10 for defending against the DLG attack and GS attack. Without sacrificing accuracy, the results demonstrate that our proposed defense can increase the mean squared error between the reconstructed data and the raw data by as much as 160$\times$ for both DLG attack and GS attack, compared with baseline defense methods. Therefore, the privacy of the FL system is significantly improved.
