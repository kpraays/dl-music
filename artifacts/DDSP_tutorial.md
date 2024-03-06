# Tutorial for Learning DDSP Framework

## Basics

- VERY IMPORTANT: Read the original DDSP paper
  - [ICLR paper link](https://openreview.net/forum?id=B1x1ma4tDr)
  - Audio examples of the paper: [supplement link](https://storage.googleapis.com/ddsp/index.html)
- Learn the **DDSP tutorial**: [Introduction to DDSP for Audio Synthesis](https://intro2ddsp.github.io/intro.html)
  - if you are interested in the background of neural audio synthesis: [A Brief History](https://intro2ddsp.github.io/background/neural-audio-synthesis.html)
  - if you don't know what is DDSP yet: [What is DDSP?](https://intro2ddsp.github.io/background/what-is-ddsp.html)
  - if you want to know what DDSP can do: [Tasks and Applications](https://intro2ddsp.github.io/background/tasks_applications.html)
  - the first simple differentiable digital audio signal processor: [A Differentiable Gain Control](https://intro2ddsp.github.io/first-steps/diff_gain.html)
  - **VERY IMPORTANT**: implement **differentiable synthesizer** in PyTorch
    - [Digital Synthesizer Modelling](https://intro2ddsp.github.io/synths/introduction.html)
    - [Writing an Oscillator in PyTorch](https://intro2ddsp.github.io/synths/oscillator.html)
    - [Additive Synthesis](https://intro2ddsp.github.io/synths/additive.html)
    - [Optimizing a Harmonic Synthesizer](https://intro2ddsp.github.io/synths/harmonic_optimize.html)
    - [Harmonic Synthesis Results](https://intro2ddsp.github.io/synths/harmonic_results.html)
    - [Differentiable Synthesis Libraries](https://intro2ddsp.github.io/synths/libraries.html)
  - **IMPORTANT**: implement **differentiable filters** in PyTorch
    - [Digital Filter Modelling](https://intro2ddsp.github.io/filters/index.html)
    - [Finite Impulse Response](https://intro2ddsp.github.io/filters/fir-intro.html)
    - [Differentiable FIR Filters](https://intro2ddsp.github.io/filters/fir-optim.html)
    - [Infinite Impulse Response (IIR) Systems](https://intro2ddsp.github.io/filters/iir_intro.html)
    - [Differentiable Implementation of IIR Filters](https://intro2ddsp.github.io/filters/iir_impl.html)
    - [Implementing Differentiable IIR in PyTorch](https://intro2ddsp.github.io/filters/iir_torch.html)
- Read/try to run the code: a PyTorch implementation of original DDSP framework [GitHub link](https://github.com/acids-ircam/ddsp_pytorch)

## Extra: Inspirations for Model Design

- A paper about **drum synthesis** using high-level timbre descriptors (brightness, depth, warmth) for control the synthesis (Lavault et al. 2022): [paper link](https://www.dafx.de/paper-archive/2022/papers/DAFx20in22_paper_20.pdf)
- A paper about **find the synthesizer parameters** from an input audio signal (Masuda & Saito 2021): [paper link](https://archives.ismir.net/ismir2021/paper/000053.pdf)
- A paper about **piano synthesis** by incorporating physical knowledge about the piano to the neural network design: [paper link](https://www.aes.org/e-lib/browse.cfm?elib=22231)
- A paper about transferring the styles of audio effects from one recording to another (Steinmetz et al. 2022): [paper link](https://arxiv.org/abs/2207.08759)
	- a 44-minutes presentation video: [YouTube link](https://www.youtube.com/watch?v=-ezTdjRpAvw)
- A **detailed review** of DDSP in music and speech synthesis (Hayes et al. 2024): [arXiv link](https://arxiv.org/abs/2308.15422), [frontiers link](https://www.frontiersin.org/articles/10.3389/frsip.2023.1284100/full)
