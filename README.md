# Introduction

In this repo you will find a batch-friendly of [Kamper et al. Segmental K-means](https://arxiv.org/abs/1703.08135). Concretely, we (re)coded their [original model](https://github.com/kamperh/bucktsong_eskmeans) so it does not require to load and compute the embedding of all segments in advance. This provides two advantatges. First, you can train the model more data than the one that fits in your memory. Second, because pooled embeddings are not pre-computed anymore, you can learn them during training time. The code gives the same exact (ie., unit test level) results as Kamper et al. when setting it accordingly (see below).

However, as I like to remind to myself ["There is No Free Lunch"](https://en.wikipedia.org/wiki/There_ain%27t_no_such_thing_as_a_free_lunch) -- as it always happens in engineering. This implementation makes the training speed a bit slower. **We are currently working on improving this**.

# Folder

- `./data/` contains all needed data for the unit tests. Concretely, it contain all intermediate representations learned in the original implementation from Kamper et al (centroids, centroids to sample from when empty, segmentations, unsupervised transcriptions) for every epoch. It is only done for Mandarin.
    - `kamperetal_epochs_centroids`
    - `kamperetal_init_centroids`
    - `kamperetal_init_segments`
    - `kamperetal_segmentation`
    - `kamperetal_transcripts`
    - `dummy`
    - `wavs`

# How to Run


# Code Structure
