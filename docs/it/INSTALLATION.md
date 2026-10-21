# Installazione

## Un solo ambiente

Installa Python 3.10+ e una versione di PyTorch per CPU/CUDA/macOS appropriata, quindi esegui `python -m pip install -e ".[acoustic,vocoder]"` dalla directory del progetto. Il comando da console `adversarial-tts`, `python -m tts` e `python inference.py` dal checkout espongono la stessa CLI. I pacchetti wheel includono anche i processi acustico e del vocoder e le risorse necessarie di sorgenti/configurazione/dizionari.

I requisiti per l'inferenza acustica sono in `requirements/acoustic-inference.txt`. Includono limiti di versione moderni e supportati per PyTorch, NumPy, SciPy, librosa, matplotlib, PyYAML, pypinyin, inflect, Unidecode, g2p-en e NLTK. Escludono i pacchetti di preprocessing usati solo per l'addestramento e gli encoder del parlante facoltativi. Il file storico `acoustic/requirements.txt` descrive uno stack di addestramento più vecchio e non è la procedura di installazione per l'inferenza.

Se una parola inglese è assente dal dizionario, g2p-en richiede cmudict e i tagger a percettrone di NLTK. Installali esplicitamente con `python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng`. Le parole presenti nel dizionario e l'elaborazione del dizionario mandarino non richiedono tale download. Il nuovo frontend non scarica automaticamente alcuna risorsa.

## Ambienti separati

L'orchestratore esegue la predizione acustica e la vocodifica in processi separati, quindi è possibile usare ambienti diversi:

```bash
python3.12 -m venv .venv-acoustic
.venv-acoustic/bin/python -m pip install -r requirements/acoustic-inference.txt
python3.12 -m venv .venv-vocoder
.venv-vocoder/bin/python -m pip install -r vocoder/requirements/inference-macos.txt
python3.12 -m venv .venv-cli
.venv-cli/bin/python -m pip install -e .

.venv-cli/bin/python inference.py --config configs/my-models.yaml --text "Hello." \
  --reference-embedding /path/to/speaker.npy --reference-mel /path/to/reference.npy \
  --output-dir outputs/example --acoustic-python .venv-acoustic/bin/python \
  --vocoder-python .venv-vocoder/bin/python
```

`ACOUSTIC_PYTHON` e `VOCODER_PYTHON` forniscono i valori predefiniti dell'ambiente; le opzioni dell'interprete sulla riga di comando hanno la precedenza. Ogni variabile indica un singolo eseguibile, non un comando shell. I percorsi relativi degli interpreti vengono resi assoluti rispetto alla directory del chiamante senza risolvere i collegamenti simbolici dei venv. I processi figli rimuovono `PYTHONPATH` e vengono eseguiti nella directory del proprio componente per isolare i moduli con nomi comuni.

La predizione acustica supporta `cpu` (predefinito), `cuda` o `auto` (CUDA se disponibile, altrimenti CPU). Il vocoder supporta `cpu`, `cuda`, `mps` e `auto` (CUDA, altrimenti CPU). La selezione di hardware non disponibile produce un errore; il programma non cambia silenziosamente un dispositivo selezionato esplicitamente.

## Encoder del parlante facoltativo

Gli embedding esistenti a 512 dimensioni non richiedono dipendenze aggiuntive per l'encoder. Per generarne uno dall'audio di riferimento, installa `requirements/speaker-encoder.txt` nell'ambiente acustico e passa esplicitamente `--speaker-encoder pyannote --reference-audio ...` al posto di una coppia embedding/mel di riferimento. La famiglia di condizionamento conservata usa vettori a 512 dimensioni nello stile di `pyannote/embedding`. L'encoder può richiedere l'approvazione di accesso di Hugging Face e `HF_TOKEN`; fornisci il token privatamente tramite l'ambiente. Nei sorgenti o nelle configurazioni non è incluso alcun token. Consulta la [scheda del modello dell'encoder](https://huggingface.co/pyannote/embedding) per le sue condizioni di accesso e utilizzo.

I download facoltativi di modelli avvengono solo quando viene selezionato tale encoder. L'opzione di generazione degli embedding non è stata testata con un checkpoint dell'encoder scaricato nel registro dei test locali; il percorso con embedding precalcolati viene verificato dall'inizio alla fine.
