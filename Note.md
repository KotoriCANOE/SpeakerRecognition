# Speaker Recognition Note

## 01.1

plain network with ResBlocks in between
- won't converge
- gradient vanishing: the closer to the input, the lower the gradients

## 01.2

added BatchNorm to ResBlocks and EBlocks
- converge
- overfitting: training loss decreased, but validation loss fluctuated, when learning rate > 4e-4

## 02.1

added skip connection between Eblocks
- hard to converge
- validation loss fluctuation

## 02.2

removed BatchNorm in EBlocks
- converge
- overfitting: validation loss fluctuated after 15k steps and learning rate > 3e-4

## 03

added skip connection to FCBlock
correctly calculate accuracy
- overfitting: learning rate > 5e-4

## 04

(furthur) replace avg pooling with max pooling
- resolved overfitting: validation loss won't fluctuate
- higher loss, lower accuracy

## 05

(futhur) learning rate: 1e-3 => 4e-4

## 06

added dropout=0.5 before the last dense layer

## 07

(unchanged) removed 1 dense layer in FCBlock
- performance drop significantly

## 08

(unchanged) FCBlock/dense1: number of features 512 => 1024
- performance drop
- overfitting: validation loss fluctuated

## 09

added SEUnit to residuals in ResBlocks
- a bit lower loss, a bit higher accuracy
- less validation loss fluctuation

## 10

added SEUnit to residuals in EBlocks
- significantly improved loss and accuracy

## 11

(unchanged) added SEUnit after first conv in EBlocks
- slightly performance drop

## 12

Validation set size: 32 => 256
