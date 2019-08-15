---
output: html_document
bibliography: [../references/references.bib]
suppress-bibliography: true
---

## Related work

### CausalGAN

@kocaoglu2018causalgan present a method to separate the work of generating images from the work of learning a causal model. They assume that the true causal graph is given, and train one GAN to sample plausible configurations of features from that causal model and another GAN to generate images given different configurations of features. This separation allows them to sample implausible but imaginable images (e.g. women with mustaches) by *intervention* on a labeled variable (e.g. *mustache*) in the causal model, while still being able to sample only plausible images (e.g. men with mustaches) by *conditioning* on the same variable.
