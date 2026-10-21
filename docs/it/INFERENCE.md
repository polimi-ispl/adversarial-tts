# Inferenza

Esegui `python inference.py --help`, `python -m tts --help` o `adversarial-tts --help` per la stessa interfaccia da testo a forma d'onda.

## Input

`--config` seleziona un file YAML del bundle di modelli. `--vocoder V|VR|VRT` seleziona la relativa voce del vocoder. Fornisci `--text` oppure `--text-file`; quest'ultimo è UTF-8, con una riga non vuota per frase. Il testo viene normalizzato con le regole di pronuncia bilingui e mappato al vocabolario originale dei simboli del modello acustico.

Il condizionamento richiede sia un embedding del parlante sia un riferimento dell'enunciato:

- Usa `--reference-embedding` con un array `.npy` numerico di forma `[512]` o `[1,512]`.
- Usa `--reference-mel` con un log mel `[80,T]`, oppure aggiungi `--reference-mel-layout TC` per `[T,80]`.
- In alternativa, `--reference-audio` estrae il mel dell'enunciato usando il frontend acustico configurato alla sua frequenza di campionamento.
- Per generare un nuovo embedding da quell'audio, ometti `--reference-embedding` e aggiungi esplicitamente `--speaker-encoder pyannote`, dopo aver installato le dipendenze facoltative e soddisfatto i requisiti di accesso. Non passare contemporaneamente un embedding precalcolato e un encoder del parlante.

Usa preprocessing del riferimento e convenzioni dell'encoder del parlante corrispondenti al tuo checkpoint acustico. Questo runtime non richiede un formato per i nomi dei file di riferimento, una ricerca tramite ID del parlante, una mappa modificabile dei parlanti o una directory fissa del dataset.

`--pitch-control`, `--energy-control` e `--duration-control` sono controlli finiti e positivi, ciascuno con valore predefinito 1.0. Vengono passati al modello acustico senza modificare le formule di addestramento.

## Flusso dei dati e output

Il processo acustico carica `checkpoint['model']` in modo rigoroso, esegue il suo metodo `.inference()`, acquisisce l'output postnet `[B,T,80]` e le lunghezze effettive predette dei mel, ritaglia ogni output e scrive `[80,true_T]`. I valori non vengono nuovamente normalizzati, traslati o passati attraverso un frontend per forme d'onda. Ogni frase viene gestita separatamente, quindi i frame di padding di un'altra frase non possono finire nella sua forma d'onda.

Il caricatore BigVGAN al livello principale legge il `checkpoint['generator']` selezionato e il JSON corrispondente. La frequenza di campionamento, la forma mono e il numero di frame della forma d'onda vengono controllati prima della pubblicazione finale. `--output-dir` riceve `speech.wav` per `--text`, oppure WAV numerati per `--text-file`, più `synthesis.json` con lunghezze, conteggi dei token e percorsi dei checkpoint selezionati. `--save-mels` conserva inoltre le predizioni senza padding sotto `mels/`.

I file di output esistenti richiedono `--overwrite`. Gli output che sono directory o collegamenti simbolici vengono rifiutati. Il programma rifiuta un percorso di output che identifica la stessa risorsa di qualsiasi bundle di input, checkpoint/configurazione/statistiche/dizionario selezionato, file di testo, embedding, mel di riferimento o audio di riferimento, anche con `--overwrite`. Il calcolo dei processi avviene in uno spazio di lavoro temporaneo; i file convalidati vengono pubblicati atomicamente, un file alla volta. La pubblicazione di più file non è un'operazione transazionale del filesystem.

## Altri punti di ingresso per la sintesi

`bash scripts/vocoder.sh mel ...` esegue la sintesi autonoma da mel salvati; `wav ...` esegue analisi/risintesi da input WAV. Entrambi accettano `--config`, `--checkpoint_file` e `--device` espliciti e corrispondenti, conservano i suffissi `_generated_e2e.wav`/`_generated.wav` e saltano gli output completati, a meno che non venga passato `--overwrite`. I launcher alla radice rendono i percorsi CLI supportati relativi al chiamante prima di cambiare la directory di lavoro nel componente.

`scripts/mel_bridge.py convert` gestisce i layout espliciti CT/TC/BCT/BTC dei file senza cambiare la scala, con lunghezze effettive obbligatorie per gli array in batch. `inspect` confronta i campi di preprocessing e segnala le differenze note tra i frontend. Questo strumento per file è facoltativo per gli array salvati esternamente; la sintesi unificata da testo non richiede un comando di conversione separato.
