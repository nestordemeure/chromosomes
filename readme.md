# Chromotypage

Using deep-learning and the fast.ai library to classify chromosome pictures.

## Process

- I load the pictures and take their negativ (let me fill spaces with black during data augmentation)
- I split into a 80%/20% training/testing set
- I deform the pictures so that they fit into a 224x224 square
- I use data augmentation on the training set (playig with scale, vertical flip, rotation, etc)
- I normalize the pictures (mean/std)
- I train a resnet34 (using the structure but not reusing weights) with a LabelSmoothingCrossEntropy loss (adapted when labels can be noisy)

## Things that did not work

### Failures

- doing augmentation with horizontal flipping had a very bad effect on the training
- resizing pictures by adding properly sized borders so that they are not deformed degraded the learning (i am unsure why)
- larger batch size reduce computing time but seem to also reduce accuracy (default works well)

### No effect

- using a jpg or png format does not seem to impact the accuracy
- improving the contrast of the pictures before training had no effect (the network is proably able to do that without my help)
- learning with increasing picture size (64->128->224) seem not to be helpful as the scores did not transfer from a size to another (this might not be true anymore) but learning at a reduced size goes much faster with little impact on the accuracy
- transfer learning with the reset weights seem to not be useful (it makes sense as the pictures look nothing like traditional photographs)
- resnet underfits while xresnet and mxresnet overfits but, otherwise, it is unclear wether xresnet is better
- the Ranger optimizer had no visible effect

## Todo

- try a larger resnet
- build a nice predict function
- update to fast.ai V2
