---
output: html_document
bibliography: [../references/references.bib]
suppress-bibliography: true
---

## Experiments

### Negative Control

Augmenting the GAN would be useless for improving the "perception" of the classifier if the classifier already has an easy time distinguishing the causal digit from the noncausal one. So we first show that we chose an appropriately difficult nonlinear transform for our dataset.

Classifier performance: {{CLASSIFIER PERFORMANCE ON ACTUAL TARGET IMAGES (low)}}

### Positive Control

The GAN isn't going to learn to represent the correct counterfactual images unless having the correct counterfactual images would be useful. So we generate an "oracle" set of augmented images with the correct counterfactuals, given our causal model of the data. We show that the linear classifier can use the correct counterfactuals to overcome its difficulty distinguishing the "causal" from "noncausal" target images.

Classifier performance: {{CLASSIFIER PERFORMANCE ON AUGMENTED TARGET IMAGES (high)}}

### Regular GAN

We train a GAN and linear classifier together on the actual images and their labels.

We show that classifier performance is still low.

Classifier performance: {{CLASSIFIER PERFORMANCE ON ACTUAL TARGET IMAGES WHEN TRAINED WITH JOINT GAN+CLASSIFIER OBJECTIVE (low)}}

QUESTION: do the images look good?

QUESTION: inspect probabilities and CFs:

* What is P(3, 4)? How do these compare to the correct probabilities?
  * CORRECT: P(3 AND 4) = 0.72; MODEL: P(3 AND 4) = ;
  * CORRECT: P(3 AND !4) = 0.02; MODEL: P(3 AND !4) = ;
  * CORRECT: P(!3 AND 4) = 0.24; MODEL: P(3 AND 4) = ;
  * CORRECT: P(!3 AND !4) = 0.02; MODEL: P(3 AND !4) = ;
* Inspect CFs (after adding CF transform):
  * (how) are the CFs different for "causal" and "noncausal" target images?
  * Given actual "causal" and "noncausal" target images, what is P_CF(3 | !4)? How does this compare to the correct CF probabilities?
    * "causal" target (C, cE): CORRECT: P_CF(3 | !4) = ; MODEL: P(3 | !4) = ;
    * "noncausal" target (nC, bE): CORRECT: P_CF(3 | !4) = ; MODEL: P(3 | !4) = ;

### Causal GAN

We train ALI and a linear classifier together on the actual images and their labels. We augment the linear classifier's input to include the CF transformed images.

We show that classifier performance improves (hopefully).

Classifier performance: {{CLASSIFIER PERFORMANCE ON ACTUAL TARGET IMAGES WHEN TRAINED WITH JOINT ALI+CFTRANSFORM+CLASSIFIER OBJECTIVE (high)}}

QUESTION: do the actual images look good?

QUESTION: inspect probabilities and CFs:

* What is P(3)? What is P(4)? What is P(3 and 4)? How do these compare to the correct probabilities?
  * CORRECT: P(3) = ; MODEL: P(3) = ;
  * CORRECT: P(4) = ; MODEL: P(4) = ;
  * CORRECT: P(3 AND 4) = ; MODEL: P(3 AND 4) = ;
* Inspect CFs (after adding CF transform):
  * (how) are the CFs different for "causal" and "noncausal" target images?
  * Given actual "causal" and "noncausal" target images, what is P_CF(3 | !4)? How does this compare to the correct CF probabilities?
    * CORRECT: P_CF(3 | !4) = ; MODEL: P(3 | !4) = ;
