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

---

## 100

used Triplet Loss (batch_hard)

## 101

EBlocks/channels: 32, 32, 32, 32, 32, 32, 32
out-channels: 128

## 102

used Triplet Loss (batch_all)

## 103

EBlocks/channels: 32, 32, 32, 48, 48, 48, 48, 64, 64, 64
EBlocks/ResBlocks: 0, 1, 1, 2, 2, 2, 3, 3, 3, 3
out-channels: 256

## 104

Triplet Loss - margin: 0.5 => 1.0

## 105

Triplet Loss - margin: 1.0 => 2.0

## 106

out-channels: 64

## 107

(unchanged) out-channels: 128

## 108

out-channels: 64
Used batch_semi_hard

## 109

out-channels: 128
Used batch_semi_hard

## 110

retrained from 106 using batch_hard
out-channels: 64

## 111

retrained from 105 using batch_hard
out-channels: 256

## 112
## 113

FCBlock: Embeddings
OutBlock: One-hot labels
FCBlock: act -> dense -> act -> dense
OutBlock: act -> dense

## 114

(unchanged) OutBlock: act -> dense -> act -> dense

## 115

(unchanged) OutBlock: dense -> act -> dense
- almost the same fraction as #114, but higher classification accuracy

## 116

OutBlock: dense

## 117

(unchanged) OutBlock dropout: 0.5

## 118

(unchanged) OutBlock dropout: 0.9

## 119

(unchanged) FCBlock: added BatchNorm before embeddings

## 120

(unchanged)
FCBlock: used LayerNorm before embeddings
triplet-margin: 2.0

## 121

(unchanged)
FCBlock: used LayerNorm before embeddings
triplet-margin: 0.5

## 122

(furthur)
FCBlock: used L2 norm before embeddings
triplet-margin: 0.5
removed cross entropy loss

## 123

(unchanged)
FCBlock: used L2 norm before embeddings
triplet-margin: 0.5
added cross entropy loss

## 124

(unchanged) used batch_hard

## 125

based on ## 116
added DataPP

## 126

(unchanged)
based on ## 122
added DataPP

## 127

(unchanged)
Batch Hard with soft margin

## 128

(bug in code, re-investigation required)
Batch All without negative distance

## 129

Batch All Triplet
triplet margin: 0.5
sample duration: 2000ms => 1000ms
val size: 256 => 576
batch size: 32 => 72
group size: 2 => 4

## 130

Center Loss

## 131

Batch All Triplet
embed size: 512 => 64

## 132

(unchanged)
Center Loss
embed size: 512 => 64

##

use packed data to speed up training
pp-smooth: 0.1 => 0
pp-amplitude: 20 => 0
noise_std: 0.025 => 0.01

## 133

[model2] Dense Connection
InBlock-EBlock/channels: 32, 16, 16, 16, 16, 16, 16

## 134

[model1] Residual Connection
InBlock-EBlock/channels: 16, 32, 48, 64, 80, 96, 112, 128

## 135

[model2]
InBlock-EBlock/channels: 32, 16, 16, 24, 24, 32, 32, 40, 40
InBlock-EBlock/ResBlocks: 0, 0, 1, 1, 2, 2, 2, 2, 2

## 136

[model1]
Center Loss: fixed bias, decay=0.9

## 138

(unchanged)
[model2]
Triplet Loss
InBlock-EBlock/channels: 32, 32, 32, 32, 32, 32, 32, 32
InBlock-EBlock/ResBlocks: 0, 0, 1, 1, 2, 2, 3, 3

## 137

[model2]
Triplet Loss
InBlock-EBlock/channels: 32, 32, 32, 40, 40, 48, 48, 56, 56, 64, 64
InBlock-EBlock/ResBlocks: 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3

## 139

Center Loss: decay=0.5

## 140

Center Loss: decay=0.9

## 141

(chosen)
Center Loss: decay=0.95

## 142

Center Loss: decay=0.99

## 143

Triplet Loss
(original) lr=1e-3, t_mul=2.0, m_mul=1.0, alpha=0
SGDR: lr=4e-3, t_mul=2.0, m_mul=0.75, alpha=1e-2

##

(unchanged)
Triplet Loss
SGDR: lr=8e-3, t_mul=2.0, m_mul=0.75, alpha=1e-2

## 144

Triplet Loss
SGDR: lr=1.4e-3, t_mul=2.0, m_mul=0.9, alpha=1e-2

## 145

(unchanged)
revert to previous DataPP

## 146

Center Loss (decay=0.95)

## 147

Triplet Loss
embed size: 64
SGDR: lr=1e-3, t_mul=2.0, m_mul=1.0, alpha=1e-1
exponential decay: step=1000, rate=0.999

## 148

Center Loss (decay=0.95)

## 149

Triplet Loss
embed size: 512

## 150

embed size: 4096

## 151

new naming
embed size: 512

## 152

normalization: None
DenseNet

## 153

normalization: None
origin net

## 154

normalization: None
VGG-like net
vox2_test
batch size: 20
down-sampling: avg pooling

## 155

normalization: None
VGG-like net
vox2_dev
batch size: 72
down-sampling: avg pooling

## 156

down-sampling: strided convolution

## 157

down-sampling: max pooling

## 158

down-sampling: avg pooling
(same as ##154)
