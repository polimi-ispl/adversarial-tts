# Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis

di Jiayan Cui, Davide Salvi, Paolo Bestagini, and Stefano Tubaro.

Implementazione e strumenti di inferenza per *Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis*, accettato a **IEEE WIFS 2026**.

<p align="center">
  <a href="https://github.com/polimi-ispl/adversarial-tts/blob/main/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper PDF"></a>
  <a href="https://cocii.github.io/adversarial-tts/"><img src="https://img.shields.io/badge/Audio-Samples-1f6feb.svg" alt="Audio samples"></a>
  <a href="https://huggingface.co/Cocii/adversarial-tts/tree/main"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffcc00.svg" alt="Model checkpoints"></a>
</p>

[English](README.md) · [简体中文](README.zh-CN.md) · **Italiano**

## Campioni audio

Confronta sei voci, tre testi e i vocoder di base e guidati dai rilevatori nella **[pagina di ascolto interattiva](https://cocii.github.io/adversarial-tts/)**. Il lettore consente di mantenere la posizione durante il confronto. È disponibile anche la [raccolta audio originale](https://my.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh).

[Panoramica](#panoramica) · [Metodo](#metodo) · [Risultati](#risultati) · [Avvio rapido](#avvio-rapido) · [Documentazione](#documentazione)

## Panoramica

Proponiamo un sistema Text-to-Speech che genera parlato **intrinsecamente avversario** ai rilevatori di deepfake vocali. Invece di applicare perturbazioni o rumore dopo la sintesi, eseguiamo il **fine-tuning del vocoder** di un sistema TTS a due stadi con uno o più rilevatori congelati nel ciclo di addestramento. Non servono post-processing né costi aggiuntivi durante l'inferenza.

Il repository riunisce il metodo di ricerca, i modelli preaddestrati e un'interfaccia di inferenza condizionata da riferimenti:

- **Tre vocoder confrontabili:** base `V`, `VR` guidato da RawNet2 e `VRT` guidato congiuntamente, con un unico modello acustico.
- **Valutazione del trasferimento:** AASIST è escluso dal fine-tuning e misura il trasferimento a un rilevatore mai visto.
- **Risorse di ricerca utilizzabili:** checkpoint scaricabili, confronto audio interattivo e un'interfaccia a riga di comando per la sintesi.

## Metodo

### Addestramento e inferenza

```text
Testo → modello acustico congelato → mel → vocoder BigVGAN → forma d'onda
                                              ↑                │
                         aggiornamento vocoder └─ loss rilevatori┘
                                                (solo training)
```

Durante il fine-tuning, RawNet2 e/o TSSD congelati forniscono un feedback differenziabile. I pesi del modello acustico e dei rilevatori restano fissi. In inferenza, il vocoder selezionato genera direttamente l'audio senza eseguire i rilevatori.

I nomi CLI `V`, `VR` e `VRT` corrispondono rispettivamente alle varianti di base, guidata da RawNet2 e guidata da RawNet2 + TSSD dell'articolo.

### Obiettivo

Il generatore comprende un modello acustico $`\mathcal{A}`$ (testo → mel-spettrogramma) e un vocoder $`\mathcal{V}`$ (mel-spettrogramma → forma d'onda). Manteniamo fisso $`\mathcal{A}`$ e aggiorniamo solo $`\mathcal{V}`$ con l'obiettivo:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\mathcal{V}} + \lambda \, \mathcal{L}_{\mathcal{D}}
```

$`\mathcal{L}_{\mathcal{V}}`$ è la perdita originale del vocoder, mentre $`\mathcal{L}_{\mathcal{D}}`$ è l'entropia incrociata binaria dei rilevatori congelati rispetto alla classe autentica. Con più rilevatori:

```math
\mathcal{L}_{\mathcal{D}} = \sum_{k=1}^{K} \alpha_k \, \mathbb{E}_{\mathbf{t}} \left[ -\log\left(1 - \mathcal{D}_k(\mathcal{V}(\mathcal{A}(\mathbf{t})))\right) \right]
```

Qui $`\lambda`$ bilancia gli obiettivi di sintesi e rilevamento, mentre $`\alpha_k`$ pesa il contributo del rilevatore $`k`$ tra i $`K`$ rilevatori congelati.

I gradienti attraversano i rilevatori congelati e aggiornano solo i parametri del vocoder.

**Configurazione:** FastSpeech2 migliorato, BigVGAN e dati LibriTTS. RawNet2 e TSSD sono usati nel fine-tuning white-box; AASIST è usato solo nel test black-box.

| Vocoder | Addestramento guidato | Scopo |
| --- | --- | --- |
| `V` | Nessuno | Sintesi di base |
| `VR` | RawNet2 | Attacco a un rilevatore |
| `VRT` | RawNet2 + TSSD | Attacco a più rilevatori |

## Risultati

EER (%) riportato nell'articolo per ogni rilevatore: valori maggiori indicano un attacco più forte; il 50% corrisponde al caso casuale.

| Vocoder | RawNet2 | TSSD | AASIST (mai visto) |
| --- | --- | --- | --- |
| `V` | 5.80 | 3.40 | 0.85 |
| `VR` | 39.05 | 38.33 | 21.09 |
| `VRT` | 43.87 | 43.31 | 23.26 |

Il fine-tuning migliora anche la qualità percettiva: il FAD con embedding CLAP scende da 31.25 per `V` a 22.95 per `VR` e 17.81 per `VRT`. Un test di ascolto in stile MUSHRA con 20 esperti giudica i vocoder adattati più realistici del modello di base. L'articolo presenta la valutazione completa.

## Avvio rapido

È richiesto Python **3.10+**; i comandi usano Python 3.12. Installa una versione di PyTorch adatta al tuo hardware quando necessario.

### 1. Installa

```bash
git clone https://github.com/polimi-ispl/adversarial-tts.git
cd adversarial-tts
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[acoustic,vocoder]" huggingface_hub
```

### 2. Scarica i modelli

```bash
.venv/bin/hf download Cocii/adversarial-tts --local-dir checkpoints
```

### 3. Sintetizza

Fornisci un embedding del parlante di forma `[512]` o `[1,512]` e un mel di riferimento con canali per primi `[80, time]`. Il solo embedding non basta. Per usare un audio di riferimento, sostituisci `--reference-mel ...` con `--reference-audio /path/to/reference.wav`.

```bash
.venv/bin/python inference.py \
  --config checkpoints/tts.yaml \
  --text "Hello, this is a speech synthesis example." \
  --reference-embedding /path/to/speaker.npy \
  --reference-mel /path/to/reference-mel.npy \
  --vocoder VRT --output-dir outputs/example \
  --acoustic-device cpu --vocoder-device cpu
```

Il risultato è `outputs/example/speech.wav` con i metadati di sintesi. Seleziona `V`, `VR` o `VRT` con `--vocoder`. Su Apple Silicon supportato, il vocoder può usare `--vocoder-device mps`; il modello acustico usa CPU o CUDA.

### Output e opzioni comuni

| Elemento | Comportamento |
| --- | --- |
| `speech.wav` | Parlato mono a 22.050 Hz per un singolo testo. |
| `synthesis.json` | Metadati, lunghezze, conteggi dei token e percorsi dei checkpoint selezionati. |
| `--save-mels` | Conserva anche il mel predetto senza padding in `mels/`. |
| `--text-file sentences.txt` | Sostituisce `--text`: una frase non vuota per riga UTF-8, con WAV numerati. |
| `--pitch-control`, `--energy-control`, `--duration-control` | Controlli acustici positivi, ciascuno con valore predefinito `1.0`. |
| `--reference-mel-layout TC` | Accetta un mel `[time, 80]` invece del formato predefinito `[80, time]`. |

Il frontend supporta inglese e mandarino. Per sostituire output esistenti serve `--overwrite`. La [guida all'inferenza](docs/it/INFERENCE.md) descrive tutti gli argomenti e gli ambienti separati per modello acustico e vocoder.

<details>
<summary>Audio di riferimento e risorse del frontend inglese</summary>

Mantieni l'embedding del parlante e fornisci un WAV per il condizionamento dell'enunciato:

```bash
.venv/bin/python inference.py \
  --config checkpoints/tts.yaml \
  --text "Hello, this is a speech synthesis example." \
  --reference-embedding /path/to/speaker.npy \
  --reference-audio /path/to/reference.wav \
  --vocoder VRT --output-dir outputs/reference-audio \
  --acoustic-device cpu --vocoder-device cpu
```

Per le parole inglesi assenti dal dizionario, installa le risorse NLTK usate da `g2p-en`:

```bash
.venv/bin/python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

L'embedding rappresenta l'identità del parlante; il mel/audio di riferimento condiziona l'enunciato. Per estrarre un embedding con un encoder opzionale, consulta l'[installazione](docs/it/INSTALLATION.md).

</details>

## Modelli

La **[raccolta di modelli su Hugging Face](https://huggingface.co/Cocii/adversarial-tts/tree/main)** contiene checkpoint acustico, tre vocoder, configurazioni, dizionari e `SHA256SUMS`. Il bundle `tts.yaml` usa percorsi relativi ed è pronto per il comando di sintesi sopra.

| File | Scopo |
| --- | --- |
| `acoustic_inference.pt` | Pesi acustici per l'inferenza. |
| `g_V` | Vocoder BigVGAN di base. |
| `g_VR` | Vocoder guidato da RawNet2. |
| `g_VRT` | Vocoder guidato da RawNet2 + TSSD. |
| `tts.yaml`, `model.yaml`, `preprocess.yaml`, `config_*.json` | Bundle di inferenza e configurazioni corrispondenti. |
| `96000.pth.tar` | Checkpoint acustico originale con lo stato dell'ottimizzatore. |

I tre vocoder condividono l'interfaccia **22.050 Hz / 80 bande mel / hop 256**, con FFT/finestra 1024 e intervallo mel 0–8.000 Hz. Usa ogni checkpoint con la sua configurazione. L'inferenza unificata taglia il mel postnet alla lunghezza effettiva e lo passa al vocoder con i canali per primi.

Per un confronto controllato, mantieni invariati testo, embedding del parlante, riferimento e controlli acustici, cambiando solo `--vocoder`.

## Struttura del repository

| Percorso | Contenuto |
| --- | --- |
| [`inference.py`](inference.py), [`tts/`](tts/) | CLI unificata, validazione dei bundle e processi di inferenza. |
| [`acoustic/`](acoustic/) | Modello acustico AdaSpeech e frontend bilingue. |
| [`vocoder/`](vocoder/) | Vocoder BigVGAN, rilevatori e codice di addestramento di ricerca conservato. |
| [`configs/`](configs/) | Esempio di configurazione del bundle. |
| [`site/`](site/) | Pagina statica di ascolto e risorse audio. |
| [`tests/`](tests/), [`docs/`](docs/) | Controlli automatici e documentazione di utilizzo e ricerca. |

## Ricerca e riproducibilità

EER, FAD e test di ascolto sopra sono **risultati riportati nell'articolo**. La validazione del rilascio controlla il caricamento dei checkpoint e la sintesi eseguibile da testo a WAV. La [guida ai test](docs/it/TESTING.md) specifica la copertura; la [guida all'addestramento](docs/it/TRAINING.md) documenta le risorse dell'esperimento originale e i requisiti del trainer conservato.

## Documentazione

- [Installazione](docs/it/INSTALLATION.md)
- [Inferenza](docs/it/INFERENCE.md)
- [Modelli e condizionamento](docs/it/MODELS.md)
- [Addestramento](docs/it/TRAINING.md)
- [Test](docs/it/TESTING.md)

L'interfaccia di inferenza usa l'implementazione AdaSpeech condizionata da un riferimento e BigVGAN. Nell'articolo il modello acustico è descritto come FastSpeech2 migliorato. I controlli del runtime verificano compatibilità dei checkpoint e sintesi eseguibile; i risultati sopra sono quelli riportati nell'articolo. La guida all'addestramento descrive il trainer di ricerca conservato e le risorse richieste.

## Uso responsabile

Questo lavoro mira a evidenziare debolezze dei rilevatori di deepfake vocali e a sostenere contromisure più robuste. Le stesse tecniche potrebbero essere usate impropriamente per generare parlato sintetico più difficile da rilevare. Il lavoro è rilasciato **esclusivamente per la ricerca sulla sicurezza difensiva** e incoraggia studi su rilevamento robusto, protocolli di valutazione e linee guida etiche per l’audio generativo.

## Citazione

```bibtex
@inproceedings{cui2026detectorguided,
  title     = {Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis},
  author    = {Cui, Jiayan and Salvi, Davide and Bestagini, Paolo and Tubaro, Stefano},
  booktitle = {IEEE International Workshop on Information Forensics and Security (WIFS)},
  year      = {2026}
}
```

## Ringraziamenti

Ringraziamo gli autori di [Microsoft AdaSpeech](https://speechresearch.github.io/adaspeech/) e [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN), insieme ai contributori delle implementazioni acustiche, dei vocoder e dei rilevatori. Attribuzioni e avvisi applicabili sono conservati in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) e [LICENSE.md](LICENSE.md).
