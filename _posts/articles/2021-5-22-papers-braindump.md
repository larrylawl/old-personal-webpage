---
layout: post
title: "NLP Papers Braindump"
author: "Larry Law"
categories: articles
tags: [misc]
image: forest.jpg
---
- [Motivation](#motivation)
- [Probing](#probing)
- [Natural Logic](#natural-logic)
- [Multilingual Models](#multilingual-models)

<!-- no toc -->
## Motivation
- **Bird's eye view.** With a bird's eye view of what's useful of each paper, it's easier for my own research to build upon the work of others. 
- **Quick retrieval.** First **sorted by topics** as each topic serve a specific purpose (e.g. if my research contribution is on introducing natural logic (NL) probing, probing topic will primarly serve as related works while NL topic will serve to refine my contribution). Then **sorted by ratings** to retrieve papers which are most inspiring. Then **sorted by year** for relevance.
- **Consistent reading.** Hopefully the "accumulative" effect of this doc motivates me to stay on top of my literature, especially in a field like NLP wherein thins move so quickly.

<!-- ## Topic
`*****`
`****`
`***`
`**`
`*` -->

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

`****`

**Probing Linguistic Systematicity** (https://ai.facebook.com/research/publications/probing-linguistic-systematicity/, Emily Goodwin, 2020)
- Jabberwocky sentences
- Artificial language based on semantic relations.

**What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties** (https://www.aclweb.org/anthology/P18-1198/, Alexis Conneau, ACL2018)
- SR ranking correlation with upstream tasks
- Human bound for tasks via 1x linguistically-trained autor.
- **Surface**: sentence length, word content (similar to MLM)
- **Synthetic**: bigram shifts, tree depth, top const
- **Semantic**: Tense, subj no., obj no., semantic odd man out, coordinate inversion

**Does String-Based Neural MT Learn Source Syntax?** (https://www.aclweb.org/anthology/D16-1159/, Xing Shi, 2016)

`***`
<!-- 2020 -->

**Finding Universal Grammatical Relations in Multilingual BERT** (https://arxiv.org/pdf/2005.04511.pdf, Ethan, ACL2020)
- Extends structural probe to multi-lingual context

**Do Neural Models Learn Systematicity
of Monotonicity Inference in Natural Language?** *(https://www.aclweb.org/anthology/2020.acl-main.543.pdf, Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, and Kentaro Inui, ACL2020, 3)*

<!-- Year: 2019 -->
**What do you learn from context? Probing for sentence structure in contextualized word representations** (https://openreview.net/forum?id=SJzSgnRcKX, Ian Tenney, 2019)

**Probing Natural Language Inference Models through Semantic Fragments** (https://arxiv.org/pdf/1909.07521.pdf, Kyle Richardson† and Hai Hu‡ and Lawrence S. Moss‡ and Ashish Sabharwal†, AAAI2019)
- Very related work. Tests logic and monotonicity fragments. **Generated logic fragments via verb-argument templates** described in Salvatore, Finger, and Hirata Jr (2019). Generated monotonicity fragments via **manually encoded monotonicity information** for 14 types of quan- tifiers (every, some, no, most, at least 5, at most 4, etc.) and negators (not, without) and **generated sentences using algorithm of Hu, Chen, and Moss (2019)** using a simple regular grammar and a small lexicon of about 100 words.
- Asks "can existing models be quickly re-trained or re-purposed to be robust on these fragments (if so, does mastering a given linguistic fragment affect performance on the original task)?". Uses **inoculation by fine-tuning method**.
- Limited training sets size to 3000 "since our goal is to learn from as little data as possible". Implies that multi-lingual

**Can neural networks understand monotonicity reasoning?** (https://www.aclweb.org/anthology/W19-4804.pdf, Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, Kentaro Inui, Satoshi Sekine, Lasha Abzianidze, Johan Bos, ACL2019)
- TODO

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

`***`

Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT (https://www.aclweb.org/anthology/D19-1077, Shi Jie Wu, ACL2019)
- X-NLP tasks: NLI, doc classification, NER, PoS, dependency parsing (UD)
  
`**`

`*`

![ML research comic](https://drive.google.com/uc?export=view&id=1bNRyr_Ubi2Sd-74KzKy5tJVkJoPlRKPi)