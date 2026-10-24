# Robustness of a pretrained audio watermark detector under realistic processing

A small, reproducible evaluation of the pretrained [AudioSeal](https://github.com/facebookresearch/audioseal) 16-bit watermark generator and detector on clean read speech. It asks: **how does detection, and recovery of the embedded message, change after compression, resampling, band-limiting, noise and mixing with other speech?**

This is exploratory, educational work by one author. It is not a validated benchmark (see [Limitations](#limitations)).

## Summary of findings

All numbers below are from one run (seed 42) on 60 watermarked and 60 clean clips.

- **Compression and resampling:** all 60 watermarked clips were detected and none of the 60 clean clips was flagged at the default threshold (0.5) after MP3 encoding down to 32 kbps and after 16 → 8 → 16 kHz resampling. Message bit accuracy was 100% in these conditions.
- **Low-pass filtering (4 kHz):** detection was unaffected, but message bit accuracy fell to 92.1%.
- **Additive white noise:** detection at the default threshold fell to 35% at 20 dB SNR and to 0% at 10 dB and 0 dB. Over the same conditions, AUC stayed at 1.000 at 20 dB and 10 dB, so the detector's scores still separated watermarked from clean clips; they simply fell below the fixed 0.5 threshold. At 0 dB, separation itself degraded (AUC 0.889, 95% bootstrap CI 0.830–0.938). Message bit accuracy dropped to 95.6%, 78.6% and 55.9% at 20, 10 and 0 dB.
- **Mixing with another speaker:** detection at the default threshold was 96.7%, 83.3% and 68.3% at 10, 5 and 0 dB SNR, with AUC 1.000 and message bit accuracy of at least 97.8% throughout.

**Takeaway:** a single headline number can give a misleading picture. Under noise, a fixed-threshold detection rate and a threshold-free AUC tell very different stories, and message recovery degrades on a different schedule from detection. Evaluations should report operating-point behaviour (threshold calibration) separately from ranking performance.

## Results

| Condition | Detection at 0.5 | False positives at 0.5 | AUC | AUC 95% CI | TPR at ≤1% FPR* | Message bit accuracy |
|:--|--:|--:|--:|:--:|--:|--:|
| none | 1.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 1.000 |
| mp3_128k | 1.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 1.000 |
| mp3_64k | 1.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 1.000 |
| mp3_32k | 1.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 1.000 |
| resample_8k | 1.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 1.000 |
| lowpass_4k | 1.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 0.921 |
| noise_20dB | 0.350 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 0.956 |
| noise_10dB | 0.000 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 0.786 |
| noise_0dB | 0.000 | 0.000 | 0.889 | 0.830–0.938 | 0.517 | 0.559 |
| speech_mix_10dB | 0.967 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 0.998 |
| speech_mix_5dB | 0.833 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 0.991 |
| speech_mix_0dB | 0.683 | 0.000 | 1.000 | 1.000–1.000 | 1.000 | 0.978 |

\*The threshold for TPR at ≤1% FPR is chosen on the same scores it is evaluated on, so it is an optimistic, post-hoc figure, not a deployable operating point. Values are rounded to three decimals. Full table: [`results/results_by_attack.csv`](results/results_by_attack.csv).

![Results](results/robustness.png)

## Method

- **Data:** 60 randomly selected utterances (3–8 s, mono, 16 kHz) from LibriSpeech `dev-clean`; a disjoint set of 60 utterances is used as the interfering speech for the mixing condition.
- **Watermarking:** pretrained AudioSeal (`audioseal_wm_16bits` generator, `audioseal_detector_16bits` detector), each clip embedded with a random 16-bit message (seeded). The embedded change has a mean level of about 26.5 dB below the original signal.
- **Conditions:** every condition is applied identically to the watermarked clips and to their unwatermarked originals, so false positives are measured under the same processing as detections: no processing; MP3 at 128, 64 and 32 kbps (ffmpeg/libmp3lame); 16 → 8 → 16 kHz resampling; 4th-order Butterworth low-pass at 4 kHz; white Gaussian noise at 20, 10 and 0 dB SNR; mixing with another speech utterance at 10, 5 and 0 dB SNR.
- **Metrics:** detection rate and false-positive rate at the detector's default 0.5 threshold; ROC AUC with a 95% percentile bootstrap interval over clips (500 resamples); TPR at ≤1% FPR; mean message bit accuracy over watermarked clips.
- **Environment:** run on a CPU-only Linux machine (Google Colab); package versions are in [`results/environment.json`](results/environment.json).

## Limitations

- **Small sample.** 60 clips per class, one run per condition. TPR at 1% FPR has coarse resolution with 60 negatives.
- **Degenerate intervals.** Where scores separate the classes perfectly, the bootstrap interval collapses to 1.000–1.000 and does not express real uncertainty.
- **Post-hoc thresholds.** TPR at ≤1% FPR uses thresholds chosen on the evaluation scores; a proper study would calibrate thresholds on separate data.
- **One method, one dataset.** Pretrained AudioSeal only; clean read English speech only; no comparison with other watermarking or detection methods.
- **Simple, non-adaptive conditions.** These are ordinary signal transformations, not attacks by someone who knows the detector. There is no generative resynthesis, adversarial or removal attack, and no listening tests.
- **Narrow interference.** The mixing condition uses a single other utterance, not diverse background audio or music.
- **No audio quality evaluation** beyond the watermark-to-signal level above.

Results should not be generalised to other watermarking systems, languages, recording conditions or processing chains.

## Reproduce

1. Open [`watermark_robustness.ipynb`](watermark_robustness.ipynb) in Google Colab (GPU optional) and choose *Runtime → Run all*. The notebook downloads LibriSpeech `dev-clean` (about 330 MB) and the pretrained AudioSeal weights, and needs `ffmpeg`, which Colab provides.
2. Or run locally: `pip install -r requirements.txt`, install `ffmpeg`, then run the notebook in Jupyter.
3. To redraw the figure from the saved table: `python make_figure.py`.

The notebook in this repository is the executed version from which the results above were taken.

## Repository contents

| Path | Description |
|:--|:--|
| `watermark_robustness.ipynb` | Full experiment, executed, with outputs |
| `results/results_by_attack.csv` | Results table (three decimals) |
| `results/robustness.png` | Figure (redrawn from the CSV by `make_figure.py`) |
| `results/environment.json` | Package versions and run configuration |
| `make_figure.py` | Regenerates the figure from the CSV |
| `requirements.txt` | Python dependencies |

## Licences and data

The code in this repository is released under the MIT licence (see `LICENSE`). The LibriSpeech corpus and the AudioSeal code and pretrained weights are used under their own licences and are not redistributed here; please check those terms before reuse.

## Author

Nujumniza Shanavaz. MSc Computer Science (AI, Big Data & Cloud Computing), University of East London.
