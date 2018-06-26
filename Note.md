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

InBlock/channels: 64
EBlock/channels: 32, 32, 48, 48, 48, 48, 48, 48, 48, 48
- performance improvement

## 20

(EBlock: act -> conv -> act -> conv -> ResBlock -> SEUnit +Dense)
Activation: Swish
- significantly improved performance

## 21

(EBlock: act -> conv -> act -> conv -> act -> ResBlock -> SEUnit +Dense)
(furthur) EBlock: added activation after 2nd conv
- slight improvement (require furthur investigation)

## 22

(EBlock: conv -> act -> conv -> act -> ResBlock -> SEUnit +Dense)
(furthur) EBlock: removed activation before 1st conv
- accuracy reaches 1 after 56k steps, but is lower before that

## 23

EBlock: conv -> act -> conv -> act -> ResBlock -> SEUnit +Dense
validation size: 32 => 256

## 24

(unchanged)
EBlock: act -> conv -> act -> conv -> ResBlock -> SEUnit +Dense
validation size: 32 => 256

## 25

(unchanged)
EBlock: act -> conv -> act -> conv -> act -> ResBlock -> SEUnit +Dense
improved loss summary logging and TensorBoard visualization

## 26

EBlock: conv -> act -> conv -> act -> ResBlock -> SEUnit +Dense
InBlock: removed ResBlock

## 27

EBlocks/ResBlocks: 0, 1, 1, 2, 2, 2, 3, 3, 3, 3

## 28

InBlocks/channels: 32
EBlocks/channels: 16, 32, 32, 48, 48, 48, 64, 64, 64, 64

## 29

InBlocks/channels: 32
EBlocks/channels: 32, 32, 32, 48, 48, 48, 48, 64, 64, 64

## 30

out-channels: 256 => 512

## 31

fixed incorrect loss logging and summary

## 32

(unchanged) generator-wd: 1e-6 => 1e-4

## 33

fixed unintended behaviour in logging steps: duplicate forward/backward passes
EBlock: act -> conv -> act -> conv -> ResBlock -> SEUnit +Dense
- better performance than post-activation

## 34

(unchanged) EBlock: bn -> act -> conv -> act -> conv -> ResBlock -> SEUnit +Dense

## 35

(unchanged) EBlock: act -> conv -> bn -> act -> conv -> ResBlock -> SEUnit +Dense

## 36

(unchanged) Batch Renorm

## 37

(unchanged) generator-wd: 1e-6 => 1e-5

## 38

(unchanged) removed Batch Norm

## 39

Data: added random Gaussian noise
Data: random amplitude

## 40

Data: added smoothing
Data: added more randomness and smoothing to noise generation

## 41

Data: added multi-step noise addition

## 42

(unchanged) lr exponential decay: 250 steps 0.99 rate

