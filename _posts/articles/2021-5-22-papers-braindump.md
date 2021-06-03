---
layout: post
title: "NLP Papers Braindump"
author: "Larry Law"
categories: articles
tags: [misc]
image: forest.jpg
hidden: true
---
- [Motivation](#motivation)
- [Transformers](#transformers)
- [NLI-bias](#nli-bias)
- [Probing](#probing)
- [Natural Logic](#natural-logic)
- [Multilingual Models](#multilingual-models)
- [Datasets](#datasets)
- [Attribution Methods](#attribution-methods)
- [Topic](#topic)

<!-- no toc -->
## Motivation
- **Bird's eye view.** With a bird's eye view of what's useful of each paper, it's easier for my own research to build upon the work of others. 
- **Quick retrieval.** First **sorted by topics** as each topic serve a specific purpose (e.g. if my research contribution is on introducing natural logic (NL) probing, probing topic will primarly serve as related works while NL topic will serve to refine my contribution). Then **sorted by ratings** to retrieve papers which are most inspiring. Then **sorted by year** for relevance.
- **Consistent reading.** Hopefully the "accumulative" effect of this doc motivates me to stay on top of my literature, especially in a field like NLP wherein thins move so quickly.

## Transformers
`*****`

`****`
**Self-Attention Attribution: Interpreting Information Interactions Inside Transformer** (https://arxiv.org/abs/2004.11207, Yaru Hao, Li Dong, Furu Wei, Ke Xu, AAAI2021)

`***`
**Transformer Feed-Forward Layers Are Key-Value Memories** (https://arxiv.org/pdf/2012.14913.pdf, Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy, 2020)
- **Trigger Examples.** We first retrieve the training examples (prefixes of a sentence) most associated with a given key, that is, the input texts where the memory coefficient is highest. Note that the input is a key (part of model) instead of training example (as in knowledge neurons)
- Work rendered questionable due to https://openreview.net/pdf?id=dYeAHXnpWJ4


**Knowledge Neurons in Pretrained Transformers** (https://arxiv.org/pdf/2104.08696.pdf, Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Furu Wei)
- Why are knowledge neurons hypothesised to be in the activation neurons not the FF-key? Because in it's experiment setup, it amplifies (2)/nullifies (0) neurons by controling the activation.
- Dynamically changing gradients to identify knowledge neurons
- Use unrelated facts as baseline
- Unlike our work, theirs have a score for each individual neuron. Our work hypothesise that there's a shared neuron at work across languages, which makes the identification of them even more potent.
- Work rendered questionable due to https://openreview.net/pdf?id=dYeAHXnpWJ4

`**`

`*`


## NLI-bias

`*****`

`****`
**Annotation Artifacts in Natural Language Inference Data** (https://www.aclweb.org/anthology/N18-2017.pdf, Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, Noah A. Smith, NAACL 2018)
- SOTA LMs did not perform well on "hard" models identified by hypothesis-only baseline -> implies that SOTA LMs aren't learning logic properly.

`***`
**Avoiding the Hypothesis-Only Bias in Natural Language Inference via Ensemble Adversarial Training.** (https://arxiv.org/pdf/2004.07790.pdf, Joe Stacey, Pasquale Minervini, Haim Dubossarsky, Sebastian Riedel and Tim Rocktäschel, EMNLP2020)
- When only trained on hypothesis, can acc as high as twice the majority baseline (67% vs. 34%)
- This is possible due to hypothesis- only biases, such as the observation that **negation words (“no” or “never”) are more commonly used in contradicting hypotheses** (Gururangan et al., 2018; Poliak et al., 2018). The hypothesis sen-tence length is another example of an artefact that models can learn from, with **entailment hypotheses being, on average, shorter than either contradiction or neutral hypotheses** (Gururangan et al., 2018). Important to train only on hypothesis as a sanity check that it doesn't beat majority baseline.


`**`

`*`

<!-- Subtopic -->
## Probing
<!-- Ratings: 5 -->
<!-- Year: ... -->
<!-- Ratings: 4 -->
<!-- Year: ... -->
<!-- Ratings: 3 -->
<!-- Year: 2020 -->
`*****`

**A Structural Probe for Finding Syntax in Word Representations** ([hewitt2019structural.pdf (stanford.edu)](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf), Hewitt 2019)
- Our probe **learns a linear transformation of a word representation space such that the transformed space embeds parse trees across all sentences**. … ; equivalently, it is finding the distance on the original space that best fits the tree metrics. 

**Designing and Interpreting Probes with Control Tasks** (https://arxiv.org/abs/1909.03368, John Hewitt, Percy Liang, EMNLP2019 (Runner Up Best Paper Award).)
- RQ: **But does this mean that the representations encode linguistic structure or just that the probe has learned the linguistic task?** 
- In this paper, we propose **control tasks**, which associate word types with random outputs, to complement linguistic tasks. By construction, **these tasks can only be learned by the probe itself**. So a good probe, (one that reflects the representation), should be selective, achieving high linguistic task accuracy and low control task accuracy.
- Consider using linear/bilinear probes over simple MLP as they have lower selectivity.

`****`

**Probing Linguistic Systematicity** (https://ai.facebook.com/research/publications/probing-linguistic-systematicity/, Emily Goodwin, 2020)
- Jabberwocky sentences
- Artificial language based on semantic relations.

<!-- 2018 -->
**Breaking NLI Systems with Sentences that Require Simple Lexical Inferences** (Max Glockner, Vered Shwartz and Yoav Goldberg, https://arxiv.org/pdf/1805.02266.pdf, ACL2018)
- Created NLI task from SNLI task by changing only one lexical (e.g. champagne to wine) Showed that performance on this new stats is substantially worse than those on SNLI.
- Nice sanity check: used wikipedia bigrams to discard sentences in which the replaced word created a bigram with less than 10 occurences (e.g. *Michael Jordan* to *Michael Syria*).
- (Unfortunately) uses crowdsource workers to obtain gold labels. Uses Kappa measure to show agreement in human annotations.
- Estimated human performance uisng the method described in Gong et al. (2018)
- Performed accuracy by category: antonyms, cardinals, nationalities, drinks, antonyms, colors, ordinals, countries, rooms, materials, vegetables, instruments, planets
- Wordnet baseline! So smart.

**What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties** (https://www.aclweb.org/anthology/P18-1198/, Alexis Conneau, ACL2018)
- SR ranking correlation with upstream tasks
- Human bound for tasks via 1x linguistically-trained autor.
- **Surface**: sentence length, word content (similar to MLM)
- **Synthetic**: bigram shifts, tree depth, top const
- **Semantic**: Tense, subj no., obj no., semantic odd man out, coordinate inversion

**Does String-Based Neural MT Learn Source Syntax?** (https://www.aclweb.org/anthology/D16-1159/, Xing Shi, 2016)

`***`
<!-- 2020 -->
**Probing Pretrained Language Models for Lexical Semantics** (https://www.aclweb.org/anthology/2020.emnlp-main.586.pdf, Ivan Vulic, Edoardo M. Ponti, Robert Litschko, Goran Glavas, Anna Korhonen, EMNLP2020)
- Tested how important is context to lexical semantics by comparing 1) isolation (each word is encoded in isolation, without context) and 2) AOC-M (average over context: avg over word's encodings from M different contexts/sentences). Can consider using AOC-M.

**Neural Natural Language Inference Models Partially Embed Theories of Lexical Entailment and Negation** (https://www.aclweb.org/anthology/2020.blackboxnlp-1.16/, Atticus Geiger, Kyle Richardson, Christopher Potts, BlackboxNLPWorkshop2020)
- Only tests interaction between lexical entailment (hyponym via sub) and negation (not)
- Probes on *algorithmic-level*x

**Finding Universal Grammatical Relations in Multilingual BERT** (https://arxiv.org/pdf/2005.04511.pdf, Ethan, ACL2020)
- Extends structural probe to multi-lingual context

**Do Neural Models Learn Systematicity of Monotonicity Inference in Natural Language?** *(https://www.aclweb.org/anthology/2020.acl-main.543.pdf, Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, and Kentaro Inui, ACL2020, 3)*
- Trains on their own dataset
- Found out that model is able to predict well on unseen combinations of lexical and logical phenomena, but cannot generalize to other syntactic structures

<!-- Year: 2019 -->
**A logical-based corpus for cross-lingual evaluation** (https://arxiv.org/pdf/1905.05704.pdf, Felipe Salvatore, Marcelo Finger, Roberto Hirata Jr, 2019)
- Trained on their own dataset
- Tasks: simple negation, boolean coordination, quantification, definite description, comparatives, counting, mixed (of previous tasks)
- Data generation. Realisation of template languages (formal language). Example, from premise *("Charles has visited chile", "Joe has visited Japan")*, we can generate the contradiction *Joe didn't visit Japan* or a neutral example *(Lana didn't visit France)*.
- *Can cross-lingual transfer learning be success- fully used for the Portuguese realization of those tasks?* Fine-tuning leads to an overall improvement.
- Only handles places and people, with only two languages (English and Portugese).

**Posing Fair Generalization Tasks for Natural Language Inference** (https://www.aclweb.org/anthology/D19-1456/, Atticus Geiger, Ignacio Cases, Lauri Karttunen, Christopher Potts, EMNLP-IJCNLP2019)

**What do you learn from context? Probing for sentence structure in contextualized word representations** (https://openreview.net/forum?id=SJzSgnRcKX, Ian Tenney, 2019)

**Probing Natural Language Inference Models through Semantic Fragments** (https://arxiv.org/pdf/1909.07521.pdf, Kyle Richardson† and Hai Hu‡ and Lawrence S. Moss‡ and Ashish Sabharwal†, AAAI2019)
- Trained on SNLI/MultiNLI
- Very related work. Tests logic and monotonicity fragments. **Generated logic fragments via verb-argument templates** described in Salvatore, Finger, and Hirata Jr (2019). Generated monotonicity fragments via **manually encoded monotonicity information** for 14 types of quan- tifiers (every, some, no, most, at least 5, at most 4, etc.) and negators (not, without) and **generated sentences using algorithm of Hu, Chen, and Moss (2019)** using a simple regular grammar and a small lexicon of about 100 words.
- Asks "can existing models be quickly re-trained or re-purposed to be robust on these fragments?". On the other hand, with only a few minutes of additional fine- tuning—with a carefully selected learning rate and a novel variation of “inoculation”—a BERT-based model can master all of these logic and monotonicity fragments while retaining its performance on established NLI benchmarks.
- Limited training sets size to 3000 "since our goal is to learn from as little data as possible". Implies that multi-lingual

**Can neural networks understand monotonicity reasoning?** (https://www.aclweb.org/anthology/W19-4804.pdf, Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, Kentaro Inui, Satoshi Sekine, Lasha Abzianidze, Johan Bos, ACL2019)
- Trained with SNLI/MultiNLI
- **Dataset creation.** For human-oriented portion, use crowdsourcing for hypothesis creation (i.e. alter text) and validation task (i.e. check entailment). For linguistics-oriented dataset, they collected from other publications.
- **6 tags:** lexical knowledge, reverse, conjunction, disjunction, conditionals, negative polarity items.
- Did not account for semantic exclusion
- How is this humane! That's ~80c per hour...
> **Workers were paid US$0.04 for each question,** and each question was assigned to three workers. To collect high-quality annotation results, we im- posed ten test questions on each worker, and re- moved workers who gave more than three wrong answers. **We also set the minimum time it should take to complete each question to 200 seconds.** 


**Probing What Different NLP Tasks Teach Machines about Function Word Comprehension** (https://www.aclweb.org/anthology/S19-1026.pdf, Najoung Kim, ACL2019)

<!-- Year: 2018 -->
**Evaluating Compositionality in Sentence Embeddings** (https://arxiv.org/abs/1802.04302, Ishita, 2018)
- Permutates sets of more, less, not more than.

<!-- Year: 2015 -->
**Recursive Neural Networks Can Learn Logical Semantics** (https://arxiv.org/pdf/1406.1827.pdf, Samuel R. Bowman, Christopher Potts, and Christopher D. Manning, Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality. 2015., 3)
- Probes TreeRNNs specifically

`**`

<!-- Year: 2020 -->
**Probing the Natural Language Inference Task with Automated Reasoning Tools** (https://arxiv.org/pdf/2005.02573.pdf, Zaid, 2020, 2)
- Testing how well a machine-oriented controlled natural language (Attempto Controlled English) can be used to **parse NLI sentences**, and how well **automated theorem provers** can reason over the resulting formulae.

**What does it mean to be language-agnostic? Probing multilingual sentence encoders for typological properties** (https://arxiv.org/abs/2009.12862, Rochelle Choenni, Ekaterina Shutova, 27 Sep 2020)
- Uses WALS features of languages to generate dataset; instance is language-dependent and not sentence dependent

<!-- 2019 -->
**Multilingual Probing of Deep Pre-Trained Contextual Encoders**(https://www.aclweb.org/anthology/W19-6205/, Vinit 2019)
- Same tasks as “What Can You Cram in a single !@#$% vector?”
- Punkt tokenizer to segment into discrete sentences. Moses tokenizer for appropriate languages. UDPipe dependency parser, which is trained on Universal Dependencies treebanks.

**LINSPECTOR: Multilingual Probing Tasks for Word Representations** (https://arxiv.org/abs/1903.09442, Gozde Gul Sahin, 2019)

<!-- 2018 -->

**Assessing Composition in Sentence Vector Representations** (https://www.aclweb.org/anthology/C18-1152/, Ettinger, ACL2018)
- BoW models as sanity checks for probes of word order
- Tasks: SemRole, negation

`*`

<!-- Year: 2020 -->
**Probing Multilingual BERT for Genetic and Typological Signals** (https://www.aclweb.org/anthology/2020.coling-main.105/, Taraka, 2020, 1)



## Natural Logic
`*****`

**Modeling Semantic Containment and Exclusion in Natural Language Inference** (https://nlp.stanford.edu/~wcmac/papers/natlog-coling08.pdf, Bill MacCartney and Christopher D. Manning, ICCL 2008, —received Springer Best Paper Award—, 5)
- Introduced semantic exclusion, 7 basic entailment relations, projectivity, inference, and implicatives.

<!-- Year: ... -->
`****`

<!-- Year: 2017 -->
**Annotation Artifacts in Natural Language Inference Data** (https://www.aclweb.org/anthology/N18-2017.pdf, Suchin Gururangan⋆ ♦ Swabha Swayamdipta⋆ ♥ Omer Levy♣ Roy Schwartz♣♠ Samuel R. Bowman † Noah A. Smith, NAACL 2017, 4)
- Ensure probing do not contain such annotation artifacts (e.g. negation is correlated to contradiction)
  
<!-- Ratings: 3 -->
`***`

**Can recursive neural tensor networks learn logical reasoning?** (https://arxiv.org/pdf/1312.6192.pdf, Samuel R. Bowman, 2014, ICLR)

**A SICK cure for the evaluation of compositional distributional semantic models** (http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf, M. Marelli1 , S. Menini1,2, M. Baroni1 , L. Bentivogli2 , R. Bernardi1 , R. Zamparelli, 2012)
- SICK dataset which has **relatedness in meaning** (with a 5-point rating scale as gold score) and **entailment relation between the two elements** (with three possible gold labels: entailment, contradiction, and neutral)



<!-- Year: ... -->
<!-- Ratings: 2 -->
<!-- Year: ... -->
<!-- Ratings: 1 -->
<!-- Year: ... -->

## Multilingual Models
`*****`

`****`

**Unsupervised Cross-lingual Representation Learning at Scale** (https://arxiv.org/abs/1911.02116, Alexis Conneau, Apr 2020)
- **Curse of multilinguity:** The experiments expose a trade-off as we scale the number of languages for a fixed model capacity: more languages leads to better cross-lingual performance on low-resource languages up until a point, after which the overall performance on monolingual and cross-lingual benchmarks degrades
- X-NLP benchmarks: XNLI, NER, Cross-Lingual QA, GLUE

**Multilingual Bert** (https://github.com/google-research/bert/blob/master/multilingual.md#models)
- Zero shot is significantly poorer than translate train. Only difference between them is the additional preprocessing step of machine translation of input data (for translate train, multiNLI is translated from English to target foreign language). Suggests that it's more a language understanding problem than a logic problem.

`***`

**Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT** (https://www.aclweb.org/anthology/D19-1077, Shi Jie Wu, ACL2019)
- X-NLP tasks: NLI, doc classification, NER, PoS, dependency parsing (UD)
  
`**`

`*`

## Datasets
`*****`

`****`
**XNLI: Evaluating Cross-lingual Sentence Representations** (https://arxiv.org/abs/1809.05053, Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk, Veselin Stoyanov, EMNLP2018)
- See Figure 1. Used parallel data for sentence embedding alignment between two languages before performing inference.

`***`
**Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence** (https://arxiv.org/abs/2103.08541, Tal Schuster, Adam Fisch, Regina Barzilay, NAACL2021)
- Used wikipedia revision edits to generate contrastive dataset (i.e., they contain evidence pairs that are nearly identical in language and content, with the exception that one supports a given claim while the other does not.)
- Word-level rationales for interpretability AND unsupervised learning objective - so smart! Unsupervised learning can be used for multi-lingual too!!

**Adversarial NLI: A New Benchmark for Natural Language Understanding** (https://www.aclweb.org/anthology/2020.acl-main.441/, Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, Douwe Kiela, ACL2020)
- We intro- duce a novel human-and-model-in-the-loop dataset, consisting of three rounds that progressively in- crease in difficulty and complexity, that includes annotator-provided explanations
- Doesn't specifically show that model's are exploiting statistical patterns

`**`

`*`

## Attribution Methods
`*****`
**Rethinking the Role of Gradient-Based Attribution Methods for Model Interpretability** (Suraj Srinivas, Francois Fleuret, ICLR2021, https://openreview.net/pdf?id=dYeAHXnpWJ4)
- Challenged input-gradient methods as interpretive tools *P(Y|X)* ; re-interprets them as class-conditional generative model *P(X|Y)*.
- However, imo it doesn't invalidate knowledge neurons yet. Say Y is the output label and X is the neuron. Given that i'm identifying the neuron, wouldn't it make more sense to find the probability of the neuron GIVEN the label? Furthermore, this is still research in progress.
- Moreover, it seems to question the reason why gradient-based methods work, rather than questioning the method in itself.
  - "This paper studies why input gradients can give meaningful feature attributions even though they can be changed arbitrarily without affecting the prediction. The claim in this paper is that "the learned logits in fact represent class conditional probabilities and hence input gradients given meaningful feature attributions." generative model P(X|Y) -> give meaningful feature attribution (how much each feature X is attributed to the final output) 
- Integrated gradients use input-gradient while Geva's work uses maximal activations

**Axiomatic Attribution for Deep Networks** (https://arxiv.org/abs/1703.01365, Mukund Sundararajan, Ankur Taly, Qiqi Yan, ICML2017)
- Identify two fundamental axioms: Sensitivity and Implementation Invariance that attribution methods ought to satisfy.
- Proposed attribution method integrated gradients that satisfy both axioms. Noob-friendly explanation of integrated gradients: https://distill.pub/2020/attribution-baselines/



`****`

`***`

`**`

`*`

## Topic
`*****`

`****`

`***`

`**`

`*`

![ML research comic](https://drive.google.com/uc?export=view&id=1bNRyr_Ubi2Sd-74KzKy5tJVkJoPlRKPi)