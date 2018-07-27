# Text-to-Speech Synthesis
Voice synthesis related materials using deep learning

## Lectures & Seminars
* [Deep running (Kim Tae-hoon, 2017.11)](http://tv.naver.com/v/2292650)
  * Video released by DEVIEW 2017 for easy understanding of Tacotron
* [Everyone's Labs WaveNet Study Video (Kim Seungil, 2017.10)](https://youtu.be/GyQnex_DK2k)
  * Explain what you understand about WaveNet and the video with online discussion
* [Generative Model-Based Text-to-Speech Synthesis (Heiga Zen, 2017.02)](https://youtu.be/nsrSrYtKkT8)
  * Heiga Zen, one of the authors of the WaveNet paper, introduces TTS overall technology and WaveNet introduction video
* [Deep Running, Speak in the Voice of a Beloved Person](https://popntalk.wordpress.com/2018/03/27/deep-learning-voice-of-loved-ones/) - Popok Blog, 2018.03.27.
  * AIA Life's Campaign Video 'Last Greetings' and blog post on voice synthesis technology 
  
## Dataset
* [CMU_ARCTIC (en)](http://festvox.org/cmu_arctic/)
  * US English data set created for speech synthesis research at CMU's Language Technologies Institute
* [The LJ Speech Dataset (en)](https://keithito.com/LJ-Speech-Dataset/)
  * I'm on Keith Ito's website, but I can not find where and why
* [Blizzard 2012 (en)](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one/)
  * Data set used in a corpus-based speech synthesis challenge called Blizzard Challenge 2012
* [CSTR VCTK Corpus (en)](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
  * English Multi-speaker Corpus for CSTR Voice Cloning Toolkit
### Korean Corpus
* [KSS Dataset: Korean Single speaker Speech Dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)
  
## WaveNet
### Paper
* [WaveNet: A Generative Model for Raw Audio (2016.09)](https://arxiv.org/abs/1609.03499)
  
### Articles
* [WaveNet: A Generative Model for Raw Audio (DeepMind Blog)](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

### Source Code
* https://github.com/ibab/tensorflow-wavenet
* https://github.com/r9y9/wavenet_vocoder (PyTorch)
* https://github.com/kan-bayashi/PytorchWaveNetVocoder (PyTorch)
  * [WaveNet Vocoder Samples](https://kan-bayashi.github.io/WaveNetVocoderSamples/)

#### Multi-GPU
WaveNet takes too long to learn, so I do not seem to get the answer unless I use a multi-GPU. The related code links are summarized.
* https://github.com/nakosung/tensorflow-wavenet/tree/multigpu (Tensorflow)
  * WaveNet multi GPU 구현 버전
* https://github.com/nakosung/tensorflow-wavenet/tree/model_parallel (Tensorflow)
  * WaveNet model parallelism 구현 버전

## Fast WaveNet
### Paper
* [Fast Wavenet Generation Algorithm (2016.11)](https://arxiv.org/abs/1611.09482)

### Articles

### Source Code
* https://github.com/tomlepaine/fast-wavenet
* https://github.com/dhpollack/fast-wavenet.pytorch (PyTorch)

## Parallel WaveNet
### Paper
* [Parallel WaveNet: Fast High-Fidelity Speech Synthesis (2017.11)](https://arxiv.org/abs/1711.10433)

### Articles
* [High-fidelity speech synthesis with WaveNet (DeepMind Blog)](https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/) 
### Source Code
* https://github.com/kensun0/Parallel-Wavenet (not a complete implement)

## WaveRNN
### Paper
* [Efficient Neural Audio Synthesis (2018.02)](https://arxiv.org/abs/1802.08435)

## Deep Voice
### Paper
* [Deep Voice: Real-time Neural Text-to-Speech (2017.02)](https://arxiv.org/abs/1702.07825)

## Deep Voice 2
### Paper
* [Deep Voice 2: Multi-Speaker Neural Text-to-Speech (2017.05)](https://arxiv.org/abs/1705.08947)

## Deep Voice 3
### Paper
* [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning (2017.10)](https://arxiv.org/abs/1710.07654)

### Source Code
* https://github.com/Kyubyong/deepvoice3
* https://github.com/r9y9/deepvoice3_pytorch (PyTorch)

## Tacotron
### Paper
* [Tacotron: Towards End-to-End Speech Synthesis (2017.05)](https://arxiv.org/abs/1703.10135)

### Source Code
* https://github.com/keithito/tacotron
* https://github.com/Kyubyong/tacotron
* https://github.com/barronalex/Tacotron
* https://carpedm20.github.io/tacotron/ (Multi-speaker Tacotron in TensorFlow)
  * Multi-speaker implementation of Tactron 1 and Deep Voice 2

## Tacotron 2
### Paper
* [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (2017.12)](https://arxiv.org/abs/1712.05884)

### Articles
* [Tacotron 2: Generating Human-like Speech from Text (Google Research Blog)](https://research.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html)

### Source Code
* https://github.com/riverphoenix/tacotron2 (implemented)
* https://github.com/Rayhane-mamah/Tacotron-2 (implemented)
* https://github.com/selap91/Tacotron2 (implemented)
* https://github.com/CapstoneInha/Tacotron2-rehearsal
* https://github.com/A-Jacobson/tacotron2 (PyTorch)
* https://github.com/maozhiqiang/tacotron_cn (implementation verification required / Chinese)
* https://github.com/LGizkde/Tacotron2_Tao_Shujie (check required)
* https://github.com/ruclion/tacotron_with_style_control (Style Control)

## HybridNet
* [HybridNet: A Hybrid Neural Architecture to Speed-up Autoregressive Models (2018.02)](https://openreview.net/forum?id=rJoXrxZAZ) - Yanqi Zhou et al.
  * WaveNet is used to pull out the audio context and use the LSTM from that context to generate the following samples faster. MOS is higher than WaveNet, and audio generation speed is 2 ~ 4 times faster than the same sound quality level. (Eg 40-layer WAVENET vs. 20-layer WAVENET + 1 LSTM)

## ClariNet
* [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech (2018.07)](https://arxiv.org/abs/1807.07281) - Wei Ping et al.
  * Gaussian autoregressive WaveNet with teacher-net and Gaussian
We have minimized Regularized KL divergence for highly picked distributions using inverse autoregressive flow as student-net. 
  * Propose a text-to-wave architecture that generates end-to-end speech.

### Articles
* [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](http://research.baidu.com/Blog/index-view?id=106) - Baidu Research, 2018.07.20. 

### Demo
* [Sound demos for "ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech"](https://clarinet-demo.github.io/)

## Voice Cloning
* [ISPEECH VOICE CLONING DEMOS](https://www.ispeech.org/voice-cloning)
  * Listen to famous people's voice cloning demo

### Paper
* [Neural Voice Cloning with a Few Samples (2018.02)](https://arxiv.org/abs/1802.06006)

## Speed ​​Up Strategy
* [Fast Generation for Convolutional Autoregressive Models (2017.04)](https://arxiv.org/abs/1704.06001) - Prajit Ramachandran et al.
  * This technique was applied to Wavenet and PixelCNN ++ models, and it was said that there was a speed increase of up to 21 times and 183 times, respectively. It is important to note that the speed improvement may not be greater than expected in a real environment because it is the maximum performance improvement for a specific situation.
