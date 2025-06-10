# diffmoe


A mixture of experts (MOE) that is designed for vision transformers and diffusion transformers


diffmoe uses a batch pool of tokens, where experts are selected across an entire batch of tokens.
This contrasts with token-choice MOE, where each token independantly selects the activated expert.
The batch pool complicates training - batch statistics influence the experts selected during the
forward pass. To resolve this, diffmoe trains a capacity predictor to ensure that during evaluation,
the expert selection mimics the same expert selection dynamics as training. diffmoe is a major improvement.
Information across image patches is spread out thinly - some image patches contain vital signals and
should recieve extra compute (Experts) while other image patches are largely uninformative. diffmoe's batch
pool allows allocating more capacity to complex samples.

An example of training a DiT to generate images of MNIST is in `examples/`

### Mini DiT with a Vanilla MLP, 100 epochs, generated images, no cfg
### Final validation loss 0.041
![image](https://github.com/user-attachments/assets/083fa916-c8f5-49fa-b8f4-17b3185fcb6d)


### Mini DiT with a DiffMOE MLP, 16 experts, 100 epochs, generated images, no cfg
### Final validation loss 0.0322
![image](https://github.com/user-attachments/assets/28fd55c9-14eb-41a1-82d3-7c0a54823c82)
