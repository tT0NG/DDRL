# DDRL

Image dehazing continues to be one of the most challenging inverse problems. Deep learning methods have emerged to complement traditional model-based methods and have helped define a new state of the art in achievable dehazed image quality. However, most deep learning-based methods usually design a regression network as a black-box tool to either estimate the dehazed image and/or the physical parameters in the haze model, i.e. ambient light (A) and transmission map (t). The inverse haze model may then be used to estimate the dehazed image. In this work, we proposed a Depth-aware Dehazing using Reinforcement Learning system, denoted as DDRL. DDRL generates the dehazed image by focusing on particular scene regions at a given stage such that these regions evolve in a near-to-far progressive manner. This contrasts with most recent learning-based methods that estimate these parameters in one pass. In particular, DDRL exploits the fact that the haze is less dense near the camera and gets increasingly denser as the scene moves farther away from the camera. DDRL consists of a policy network and a dehazing (regression) network. The policy network estimates the region that the dehazing network must focus on at a given stage. A novel policy regularization term is introduced for the policy network to generate the policy sequence following the near-to-far order. 

![DDRL](https://github.com/tT0NG/DDRL/blob/master/network/drl_8.png)

### Figs
Check the results images in https://github.com/tT0NG/DDRL/tree/master/figs
