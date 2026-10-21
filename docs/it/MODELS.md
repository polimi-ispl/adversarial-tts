# Modelli e condizionamento

Scarica modello acustico, vocoder V / VR / VRT, configurazioni e dizionari da **[Hugging Face](https://huggingface.co/Cocii/adversarial-tts/tree/main)**:

```bash
python -m pip install huggingface_hub
hf download Cocii/adversarial-tts --local-dir checkpoints
```

Usa `checkpoints/tts.yaml` con la CLI di sintesi. I percorsi delle risorse sono relativi al bundle. `SHA256SUMS` contiene gli hash dei modelli e delle configurazioni. Per altri checkpoint, conserva le risorse fuori dal controllo di versione e specifica i percorsi nel tuo bundle YAML.

## Schema del bundle

Consulta `configs/tts.example.yaml`. La mappa `acoustic` richiede `checkpoint`, `preprocess`, `model` e `lexicon`; `stats` è facoltativo quando il checkpoint contiene tutti i bin addestrati. La mappa `vocoders` contiene tutte le voci V/VR/VRT disponibili, ciascuna delle quali richiede `checkpoint` e `config`. Viene caricata solo la variante selezionata. I percorsi relativi hanno come base il file YAML del bundle, non la directory di lavoro corrente della shell.

Il checkpoint acustico deve contenere il dizionario di stato `model` per l'implementazione `AdaSpeech_spkr_en_cn` condizionata da un riferimento: `spk-type: spk`, `language_speaker.separated: false`, mel/postnet a 80 canali ed embedding di riferimento a 512 dimensioni. Devono corrispondere i livelli del modello, le dimensioni nascoste, il vocabolario dei simboli, gli intervalli di pitch/energia e le chiavi del checkpoint. Questo modello accetta testo in inglese o mandarino attraverso il suo frontend bilingue; non viene sostituito silenziosamente con un'architettura monolingue/XLSR diversa.

Se fornito, `stats.json` contiene le statistiche numeriche originali di pitch ed energia attese dal checkpoint, inclusi gli estremi dei bin. In sua assenza, il runtime legge gli estremi di `variance_adaptor.pitch_bins` e `variance_adaptor.energy_bins` soltanto per soddisfare il costruttore del modello invariato in una directory temporanea privata. Il caricamento rigoroso dello stato ripristina quindi **ogni bin memorizzato**, inclusi gli intervalli non uniformi, e li verifica byte per byte. I bin mancanti, non finiti, non ordinati o di forma errata vengono rifiutati. Ciò non recupera medie/deviazioni standard di addestramento, metadati del dataset o la procedura originale di normalizzazione.

I percorsi dei dizionari sono espliciti e tutti gli input di condizionamento vengono forniti indipendentemente dai nomi dei file o dalle mappe dei parlanti. Il caricatore acustico richiede PyTorch 2.6+ e usa sempre `weights_only=True`. La sua lista limitata di costruttori scalari/dtype NumPy float32/float64 consentiti supporta i metadati dei vecchi ottimizzatori; gli oggetti pickle personalizzati arbitrari restano rifiutati. Il checkpoint originale viene letto senza modifiche e non viene usato alcun ripiego su pickle senza restrizioni.

I checkpoint V/VR/VRT contengono ciascuno `generator`. La configurazione deve identificare l'architettura esatta e la convenzione di condizionamento. Riprendere l'addestramento del vocoder richiede inoltre lo stato di discriminatore/ottimizzatore/step; un checkpoint contenente solo il generatore non è un checkpoint di addestramento completo.

## Convenzioni dei mel

La configurazione acustica predefinita usa 22050 Hz, FFT/finestra 1024, hop 256, 80 bande e 0–8000 Hz. Il vocoder selezionato deve concordare su questi campi e sul prodotto dei fattori di upsampling. I soli valori predefiniti non identificano un checkpoint addestrato.

Il frontend acustico di riferimento esaminato aggiunge padding per riflessione di n_fft/2 campioni per lato (512 con i valori predefiniti); il frontend WAV BigVGAN aggiunge padding di (n_fft-hop)/2 (384) con `center=False`. BigVGAN aggiunge inoltre 1e-9 all'interno della radice quadrata della magnitudine; il frontend acustico non lo fa. Entrambi i percorsi esaminati usano il logaritmo naturale con un limite inferiore di 1e-5, senza una normalizzazione separata di media/varianza del mel nel percorso dataset/esportazione del modello. Il preprocessing esterno può differire.

L'inferenza unificata da testo passa il **mel postnet predetto** dal modello acustico direttamente a un vocoder addestrato per quella convenzione di condizionamento. Ritaglia secondo la lunghezza effettiva predetta e traspone da BTC a CT. Non riesegue l'estrazione dalla forma d'onda, non forza l'equivalenza numerica tra il frontend di riferimento e quello del vocoder e non inventa una correzione di padding/normalizzazione. Usa configurazioni di addestramento corrispondenti e convalida separatamente la qualità dei checkpoint addestrati.

Un mel di riferimento fornisce il condizionamento dell'enunciato al modello acustico; non è l'enunciato generato. Un embedding del parlante precalcolato fornisce il condizionamento dell'identità e non sostituisce il mel di riferimento. L'audio di riferimento può fornire tale mel; pyannote, se selezionato facoltativamente, può fornire il vettore di identità corrispondente a 512 dimensioni.

## Convalida dei checkpoint dell'articolo

Prima di dichiarare la riproduzione dell'articolo, verifica i pesi esatti acustici/del vocoder/dei rilevatori e le revisioni dei sorgenti, l'ambiente, il preprocessing/le suddivisioni dei dati, la polarità delle classi dei rilevatori e le procedure di valutazione. I test di esecuzione con checkpoint casuali non sono misurazioni di qualità o rilevamento. Le licenze e le autorizzazioni per qualsiasi peso distribuito, dato di addestramento e materiale vocale/di riferimento devono essere confermate prima della pubblicazione.

La convalida locale ha verificato rigorosamente lo stato acustico completo e tutti gli stati dei generatori V/VR/VRT. Tutti e tre i vocoder hanno poi sintetizzato dallo stesso mel predetto usando pesi acustici completi, identità sintetica e audio pubblico di riferimento, con il modello acustico su CPU e vocoder MPS espliciti (ripiego su CPU disabilitato). Il checkpoint acustico per sola inferenza conserva esattamente i tensori originali del modello e omette lo stato dell'ottimizzatore. Questi controlli non dimostrano l'identità del parlante originale, il recupero delle statistiche di addestramento, la qualità, le metriche dei rilevatori o la riproduzione dell'addestramento/ripresa.
