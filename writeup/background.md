---
output: html_document
bibliography: [../references/references.bib]
suppress-bibliography: true
---

## Background

### Psychology

#### People use counterfactuals to understand "cause"

@gerstenberg_eye-tracking_2017 used eyetracking to more directly show that people spontaneously consider counterfactual outcomes while making causal judgements. They tracked eye movements while participants judged whether one billiard ball caused another to go through a gate. When asked for _causal_ judgements like this, but not when asked about details of the _outcome_ (how close the caused ball came to the edge of the gate), participants' eye movements traced the counterfactual path that the caused ball would have taken had the causing ball been absent.

#### Expectations affect counterfactual and causal judgements

When researchers discuss the role of expectations in counterfactual and causal reasoning, they typically refer to two main categories of expectations: _statistical normality_ and _prescriptive normality_. Statistical normality refers to the probabilities of what is likely to occur, whereas prescriptive normality refers to what "should" occur. Both seem to influence causal reasoning, and both concepts of normality seem to have similar effects.

People often imagine counterfactual changes that would undo a negative outcome, thinking things like "If only he had driven home a little later, he wouldn't have gotten in a car crash." @kahneman_simulation_1981 show that people are sensitive to statistical normality when deciding on an "if only..." counterfactual. They presented stories where they varied whether a character Mr. Jones went home at the usual time via an unusual route or went home at an unusually early time via his usual route. Participants were much more likely to mention counterfactuals where Mr. Jones behaved more like he usually did, choosing the route to counterfactually change when the route had been unusual, and choosing the time to  counterfactually change when the time had been unusual.


Faced with this case, participants tend to say that the professor caused the problem (Knobe and Fraser, 2008, Phillips et al., 2015).


@icard_normality_2017 summarize the ways that statistical and prescriptive normality affect causal judgements into three main categories:

* **Abnormal inflation**: When there are multiple necessary causal factors for an outcome, people will be more likely to call something a cause if it is abnormal.
* **Supersession**: When there are multiple necessary causal factors for an outcome, people will be more likely to call something a cause if the _other_ factors are normal.
* **No supersession with disjunction**: When there are multiple _sufficient_ causal factors for an outcome (i.e. any one of the factors could have produced the outcome independently), people's judgements of how likely something is to be a cause does _not_ depend on the normality of the other factors.

They show how a model of causal reasoning based on _probabilistic sampling_ of counterfactual situations would result in these observed phenomena. They further present an additional principle, predicted by their model:

* **Abnormal deflation**: When there are multiple _sufficient_ (but not necessary) causal factors for an outcome, people will be more likely to call something a cause if it is normal.

They present an experiment with animations simulating billiard balls colliding, representing either conjunctive (multiple necessary factors) or disjunctive (multiple sufficient factors) causal scenarios. As predicted, they demonstrate abnormal inflation in the conjunctive scenario and abnormal deflation in the disjunctive scenario.








 (see, e.g., Danks et al., 2014, Phillips et al., 2015, Samland et al., 2016). 

  (Roxborough & Cumby, 2009).

The natural interpretation of Bayes nets is causal: we generally include a link from one variable to another if the first has a direct causal influence on the second. In fact, the idea that people rely on representations very much like Bayes nets has been shown consistent with a wide array of data on causal learning and inference in children and adults (for a review, see Sloman & Lagnado, 2015). One of the key ideas is that Bayes nets can be used not just for ordinary probabilistic inferences such as conditionalization based on observations, but also for distinctively causal manipulations known as interventions (Pearl, 2009, Spirtes et al., 1993, Woodward, 2003). Our account will make use of this framework.

Existing research provides some support for this hypothesis. When participants are given a vignette and asked to provide a counterfactual, they are more likely to mention possibilities they regard as statistically frequent but also possibilities they regard as prescriptively good (Kahneman and Miller, 1986, McCloy and Byrne, 2000). In addition, when participants are given a counterfactual and asked to rate the degree to which it is relevant or worth considering, they are more inclined to rate a possibility as relevant to the extent that it conforms to prescriptive norms (Phillips et al., 2015). These findings provide at least some initial evidence in favor of the claim that people are drawn to consider possibilities that do not violate prescriptive norms.



It has been known for decades that actual causation judgments can be influenced by statistical norms (Hilton & Slugoski, 1986). Suppose that a person leaves a lit match on the ground and thereby starts a forest fire. In such a case, the fire would not have begun if there had been no oxygen in the atmosphere, and yet we would not ordinarily say that the oxygen caused the fire. Why is this? The answer appears to involve the fact that it is so (statistically) normal for the atmosphere to contain oxygen. Our intuitions should therefore be very different if we consider a case in which the presence of oxygen is abnormal. (Suppose that matches were struck on a regular basis but there is never a fire except on the very rare occasions when oxygen is present.) In such a case, people should be more inclined to regard the presence of oxygen as a cause.

The difference between these two cases is solely in the normality of the dice roll. The success of the dice roll is statistically normal in the first case, statistically abnormal in the second. Yet this difference actually leads to a change in the degree to which people regard the coin flip as a cause: participants were significantly less inclined to say that Alex won because of the coin flip when the dice roll was abnormal than when it was normal (Kominsky et al., 2015).



Yet, though existing work clearly shows that both statistical and prescriptive norms can lead to abnormal inflation, controversy remains regarding the explanation of this effect. Researchers have suggested that the effect might arise as a result of conversational pragmatics (Driver, 2008), motivational bias (Alicke et al., 2011), relativity to frameworks (Strevens, 2013), responsibility attributions (Sytsma, Livengood, & Rose, 2012), or people’s understanding of the question (Samland & Waldmann, 2016). Here, we will be exploring a general approach that has been defended by a number of researchers in recent years, namely, that abnormal inflation reflects a process in which certain counterfactuals are treated as in some way more relevant than others (Blanchard and Schaffer, 2016, Halpern and Hitchcock, 2015, Knobe, 2010, Phillips et al., 2015).



For example, one study looked at controversial political issues (abortion, euthanasia) and found that people who had opposing moral judgments about these issues arrived at correspondingly opposing causal judgments about people who performed the relevant actions (Cushman, Knobe, & Sinnott-Armstrong, 2008).

One intriguing phenomenon that has long been recognized is that people’s judgments of actual causation can be influenced by the degree to which they regard certain events as normal. In recent years, this effect has been explored both in experimental studies and in formal models (e.g., Halpern and Hitchcock, 2015, Kominsky et al., 2015, Phillips et al., 2015).

As a result, researchers have suggested that it might be helpful to posit a single undifferentiated notion of normality that integrates both statistical and prescriptive considerations (Halpern and Hitchcock, 2015, Kominsky et al., 2015).

For example, one might hypothesize that the impact of normality is the result of a motivational bias or of conversational pragmatics (e.g., Alicke et al., 2011, Driver, 2008, Samland and Waldmann, 2016). 

People tend to select abnormal (or unexpected) over
normal (or expected) causes (Cheng & Novick, 1991; Hesslow, 1988; Hilton & Slugoski, 1986;
Knobe & Fraser, 2008). Norms and expectations also play a critical role in how people make
causal judgments about omissions, that is, events that didn’t happen (Clarke, Shepherd,
Stigall, Waller, & Zarpentine, 2015; Gerstenberg & Tenenbaum, 2016; Henne, Pinillos, &
De Brigard, 2016; McGrath, 2005; Samland & Waldmann, 2016; Stephan, Willemsen, &
Gerstenberg, 2017; Willemsen, 2016; Wolff, Barbey, & Hausknecht, 2010; Wolff, Hausknecht,
& Holmes, 2011). At any given moment in time, an infinite number of events don’t happen,
but we normally only ever consider those that violated our expectations.

people sometimes select a normal rather than
an abnormal event as the cause of the outcome (Gavanski & Wells, 1989; Gerstenberg et
al., 2018; Icard et al., 2017; Johnson & Rips, 2015; Sytsma, Livengood, & Rose, 2012). For
example, Icard et al. (2017)

To date, there is no unified account that accurately captures all of the known patterns
of normality effects on causal selection. Some accounts predict that abnormal events are
generally favored (Hall, 2007; Halpern, 2016; Halpern & Hitchcock, 2015). Others predict
that normal events are held more responsible for positive outcomes (Johnson & Rips, 2015),
that people select normal causes for normal outcomes, and abnormal causes for abnormal
outcomes (Harinen, 2017; see also Gavanski & Wells, 1989), that it depends on the causal
structure (Icard et al., 2017; Morris et al., 2018), or that it depends on what we infer about
an actor from her actions (Gerstenberg et al., 2018).
