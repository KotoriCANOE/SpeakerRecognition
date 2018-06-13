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

(unchanged/furthur) replace avg pooling with max pooling
- resolved overfitting: validation loss won't fluctuate
- higher loss, lower accuracy

## 05

(unchanged/furthur) learning rate: 1e-3 => 4e-4

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

## 13

(unchanged/furthur) batch size: 32 => 64
- might improve performance, but slower due to bottleneck in CPU
- overfitting: training loss close to 0

## 14

(require reinvestigate)
InBlock/conv2d kernel: 1x3 => 1x7
EBlock/conv2d kernel: 1x4 => 1x7

## 15

[model2]
DenseNet
InBlock: 32
EBlock: 2*16+2*32+64*6

## 16

EBlock kernel: 1x4 => 1x3

## 17

(unchanged) EBlock: removed 1x1 bottleneck layer

## 18

(unchanged) ResBlock: removed batch norm

## 19

InBlock: 64
EBlock: 2*32+8*48
- performance improvement

## 20

Activation: Swish
- significantly improved performance

## 21

(furthur) EBlock: added activation after 2nd conv
- slight improvement (require furthur investigation)

## 22

(furthur) EBlock: removed activation before 1st conv
- accuracy reaches 1 after 56k steps, but is lower before that
