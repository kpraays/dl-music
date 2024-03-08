## Resources about DDSP

### Papers and Tutorials

- A very good **DDSP tutorial**: [Introduction to DDSP for Audio Synthesis](https://intro2ddsp.github.io/intro.html)
- The **original DDSP paper** (Engel et al. 2020): [ICLR paper link](https://openreview.net/forum?id=B1x1ma4tDr)
  Link related to the paper's work (including GitHub code, Colab demo, Colab tut...): https://magenta.tensorflow.org/ddsp
- A **detailed review** of DDSP in music and speech synthesis (Hayes et al. 2024): [arXiv link](https://arxiv.org/abs/2308.15422), [frontiers link](https://www.frontiersin.org/articles/10.3389/frsip.2023.1284100/full)
- A paper about **drum synthesis** using high-level timbre descriptors (brightness, depth, warmth) for control the synthesis (Lavault et al. 2022): [paper link](https://www.dafx.de/paper-archive/2022/papers/DAFx20in22_paper_20.pdf)
- A paper about **find the synthesizer parameters** from an input audio signal (Masuda & Saito 2021): [paper link](https://archives.ismir.net/ismir2021/paper/000053.pdf)
- A paper about **piano synthesis** by incorporating physical knowledge about the piano to the neural network design: [paper link](https://www.aes.org/e-lib/browse.cfm?elib=22231)
- A paper about transfering the styles of audio effects from one recording to another (Steinmetz et al. 2022): [paper link](https://arxiv.org/abs/2207.08759)
	- a 44-minutes presentation video: [YouTube link](https://www.youtube.com/watch?v=-ezTdjRpAvw)

### Applications and Demos

- Audio examples of the original DDSP paper: [supplement link](https://storage.googleapis.com/ddsp/index.html)
- Another demo site about applying the original DDSP architecture directly for timbre transfer: [Tone Transfer](https://sites.research.google/tonetransfer)
- Examples of another architecture for neural synthesis: [audio examples](https://anonymous84654.github.io/RAVE_anonymous/)
	- high reconstruct quality and real-time ability by adding a GAN structure (Caillon & Esling 2021) [paper link](https://arxiv.org/abs/2111.05011)
- Sound examples for finding the synthesizer parameters and re-synthesis of the sounds: [GitHub link](https://nas-fm.github.io/) 
	- (Ye et al. 2023) [paper link](https://arxiv.org/abs/2305.12868)
- Sound examples for DDSP-Piano: [supplement link](http://renault.gitlab-pages.ircam.fr/dafx22-audio/jekyll/update/2022/04/25/supplementary-materials) 
- Sound examples for "style transfer of audio effects" paper: [GitHub link](https://csteinmetz1.github.io/DeepAFx-ST/)

### Datasets

- NSynth (Engel et al. 2017): [website](https://magenta.tensorflow.org/datasets/nsynth)
	- containing 305,979 musical notes
	- contains source, instrument family, qualities, pitch, and velocity
	- only 16kHz
- Amp-Space (Naradowsky 2021): [paper link](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in21_paper_47.pdf)
	- large-scale dataset of paired audio samples
	- a source audio signal, and an output signal, the result of a timbre transformation
	- \>500 hours synthesized audio + \>50 hours real audio
	- 44.1kHz monaural sound
- synth1B1 (Turian et al. 2021): [paper link](https://www.dafx.de/paper-archive/2021/proceedings/papers/DAFx20in21_paper_34.pdf)
	- 1 billion 4-second synthesized sounds
	- paired with the synthesis parameters used to generate them
	- 44.1kHz
