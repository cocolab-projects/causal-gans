---
output: html_document
bibliography: [../references/references.bib]
toc:
    depth_from: 2
    depth_to: 4
---

# Causality in Modern Deep Generative Models

[TOC]

Humans reason and communicate about causality when learning about and interacting with the world. Most statistical models (including deep learning) are limited to measuring correlation, unable to make claims about causality between variables. In this project, we aim to augment a deep generative model such that it can represent causal relations in a human-interpretable way and guide its learning to be consistent with labeled causal relations.

We train our model on a dataset of MNIST images where the presence of one digit depends causally on the presence of another digit. We jointly train ALI and a linear classifier, which labels whether each image is an instance of a causal relation. This classifier takes as input the actual image as well as a "counterfactual" image. This counterfactual image is the result of inferring a value for the actual image in the latent space, applying a fixed set of transformations, and observing the corresponding "close" alternative images.

We aim to show whether this classifier, augmented with the latent space and counterfactual transform, can accurately distinguish between perceptually similar but causally different images.

Ideally, we would like to further show whether the model can learn to structure its generator such that the counterfactual transformation exposes the appropriate causal relations.

@import "background.md"
@import "related_work.md"

## References
