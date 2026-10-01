# Grammar Scoring Engine for Spoken English

SHL Hiring Assessment 2026, Kaggle competition.

> ## Rank 1 on the public leaderboard
> **Score 0.3278** (lower is better) · [Kaggle leaderboard](https://www.kaggle.com/competitions/shl-hiring-assessment-2026/leaderboard)

The solution is one notebook, saved with all its outputs: **[grammar_scoring.ipynb](grammar_scoring.ipynb)**. It contains the report (approach, preprocessing, architecture, evaluation), the code and the charts. If GitHub does not render it, open it on [nbviewer](https://nbviewer.org/github/AMITABHSHUKLA/spoken-grammar-scoring/blob/main/grammar_scoring.ipynb).

## The task

Each clip is a 45 to 60 second spoken answer to an open question. Human raters scored its grammar against a 1 to 5 rubric, and the label is their mean opinion score on a grid of 0.5. There are 769 labelled training clips and 216 test clips. Submissions are evaluated with RMSE and Pearson correlation.

How I read the problem: grammar is a property of *what* is said, so the score has to come from understanding the speech. But raters listen to the whole answer, and hesitation, rhythm and pronunciation move together with grammar in their judgement. The engine therefore needs both the content of the answer and how it was spoken, and it has to work for speakers, and often questions, that it has never heard.

## Challenges

| Challenge | What I found | What I did |
|---|---|---|
| **Little data for a hard task** | 769 clips are far too few to train a speech model. | Large pretrained models stay frozen and only turn each clip into numbers. Small regressors learn the score from those numbers. |
| **Speakers repeat** | Two thirds of the training clips have another answer by the same speaker, scored almost the same (spread 0.21 within a speaker, 1.01 overall). The test speakers are new. | Speakers are identified by voice similarity and kept together in the cross-validation folds. Random folds would have made the result look 0.03 RMSE better than it is. |
| **Clips without speech** | The 37 clips scored 0 contain noise, not speech, yet Whisper still "transcribes" them into fluent sentences. | A CTC speech model that hears no speech sounds in a clip flags all 37 and nothing else. They get 0 by a rule, and the regressors learn only from real answers. |
| **Speech recognition hides the errors** | Whisper tends to write what the speaker meant rather than what was said: in one clip, "these trees gives" became "these trees give". | A prompt that makes Whisper keep fillers and false starts, an LLM that counts the corrections a transcript needs, and an audio language model (Voxtral) that reads the audio directly, with no transcript in between. |
| **The test set is a different mix** | A batch of 45.06 s recordings is 14% of the training set but 47% of the test set, and it scores lower. Many of its clips answer questions that never occur in training. | Results are reported separately for this batch (cross-validated RMSE 0.51, about the same as for the 60 s clips). I expect the test error to be somewhat above the cross-validated estimate. |
| **Coarse, subjective labels** | Scores are human opinions on a grid of 0.5, and only 4 training clips score below 2. | Errors are reported by score. The lowest scores are predicted too high, and the notebook says so. |
| **Simple versus best** | A blend of nine models was only 0.006 RMSE better than a blend of three. | I kept the three models that I can explain one by one. |

## Approach

```
                      ┌──────────────────────────────┐   ┌───────────┐
                 ┌──► │ Voxtral-Mini-3B (audio LLM)  │──►│ Ridge     │──┐
                 │    │ average state of audio tokens│   └───────────┘  │
                 │    └──────────────────────────────┘                  │
┌────────────┐   │    ┌──────────────────────────────┐   ┌───────────┐  │   ┌──────────────┐
│ audio clip │───┼──► │ WavLM-large, layer 20        │──►│ SVR (RBF) │──┼──►│ non-negative │──► grammar score
│ 16 kHz     │   │    │ average over time            │   └───────────┘  │   │ linear blend │    (1 to 5)
└────────────┘   │    └──────────────────────────────┘                  │   └──────────────┘
                 │    ┌──────────────────────────────┐   ┌───────────┐  │
                 ├──► │ Whisper large-v3 transcript  │──►│ LightGBM  │──┘
                 │    │ then 123 grammar features    │   └───────────┘
                 │    └──────────────────────────────┘
                 │    ┌──────────────────────────────┐
                 └──► │ wav2vec2 CTC: any speech?    │──► if not: score 0
                      └──────────────────────────────┘
```

1. **Voxtral-Mini-3B + Ridge.** An audio language model listens to the clip with the question "How accurate and complex is the speaker's grammar?". Its hidden states over the audio (layers 12 to 14, averaged) go into a ridge regression. This is the strongest single model.
2. **WavLM-large + SVR.** Layer 20 of a self-supervised speech model, averaged over time, captures how the answer sounds: fluency, rhythm and pronunciation.
3. **Transcript features + LightGBM.** 123 readable features from the Whisper large-v3 transcript: an LLM rubric score (Qwen3.5-9B reads the transcript with the official rubric, and I take the probability-weighted average of its answer 1 to 5), the rate of edits an LLM needs to make the transcript grammatical, pause and fluency statistics, and spaCy syntax statistics.
4. **Blend and rule.** A linear regression with non-negative weights combines the three predictions (weights 0.48, 0.39 and 0.26). Clips without speech get 0.

The heavy steps ran once on Kaggle's free T4 GPUs. The notebook starts from the saved features and runs on a CPU in under a minute.

## Results on the training data

| | RMSE | Pearson |
|---|---|---|
| **Training RMSE** (in-sample) | **0.180** | 0.989 |
| Cross-validated on unseen speakers, all 769 clips | 0.493 | 0.917 |
| Cross-validated on unseen speakers, the 732 clips scored 1 to 5 | 0.505 | 0.867 |

The in-sample row is the training RMSE the task asks for, and it is optimistic by construction. The cross-validated rows are the honest estimate: every clip is predicted by models that never heard its speaker.

| Model, cross-validated on the 732 scored clips | RMSE | Pearson |
|---|---|---|
| Voxtral + Ridge | 0.535 | 0.850 |
| WavLM + SVR | 0.566 | 0.831 |
| Transcript features + LightGBM | 0.608 | 0.800 |
| **Blend of the three** | **0.505** | **0.867** |
| Always predicting the mean score | 1.014 | |

## Leaderboard

### 🥇 Rank 1 · public score 0.3278

Lower is better. [See the leaderboard on Kaggle.](https://www.kaggle.com/competitions/shl-hiring-assessment-2026/leaderboard)

## Repository

```
grammar_scoring.ipynb          report, code, evaluation and charts (saved with outputs)
extraction/                    GPU and LLM steps that turn the audio into features (run once)
  transcribe.py                Whisper large-v3 transcripts with word timestamps
  audio_features.py            WavLM hidden states; wav2vec2 CTC speech check
  voxtral_features.py          Voxtral-Mini-3B hidden states over the audio tokens
  llm_features.py              LLM rubric score and grammar-correction edit rates
  fluency_features.py          pauses, speech rate, fillers, recognition confidence
  syntax_features.py           spaCy syntax and vocabulary statistics
  build_features.py            collects everything into the features/ folder the notebook reads
requirements.txt               packages for the notebook
requirements-extraction.txt    extra packages for extraction/
```

The competition audio and labels, the transcripts, the per-clip features and the prediction file are not in the repository.

## Reproducing

1. Install the packages. The notebook needs only `requirements.txt`; the extraction scripts also need `requirements-extraction.txt` and a GPU.

   ```bash
   pip install -r requirements.txt -r requirements-extraction.txt
   python -m spacy download en_core_web_sm
   ```

2. Put the competition files in `data/`: `train.csv`, `test.csv`, `train/*.wav`, `test/*.wav`.

3. Extract the features. Outputs go to `work/`, and the last step writes `features/`.

   ```bash
   python extraction/transcribe.py --tag disfluent --prompt disfluent
   python extraction/transcribe.py --tag clean --prompt none
   python extraction/audio_features.py --what wavlm
   python extraction/audio_features.py --what ctc
   python extraction/voxtral_features.py
   python extraction/llm_features.py
   python extraction/fluency_features.py
   python extraction/syntax_features.py
   python extraction/build_features.py
   ```

   `llm_features.py` calls an OpenAI-compatible chat endpoint that returns token log-probabilities. Copy `.env.example` to `.env` and fill it in. I used the open-weights model `cyankiwi/Qwen3.5-9B-AWQ-4bit` served with vLLM; with another model the features, and so the scores, will differ somewhat.

4. Run the notebook. It needs no GPU.

   ```bash
   jupyter nbconvert --to notebook --execute --inplace grammar_scoring.ipynb
   ```
