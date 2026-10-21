# Test

Installa le dipendenze di inferenza/sviluppo, quindi esegui:

```bash
python -m pip install -e ".[acoustic,vocoder]" -r requirements/development.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
(cd vocoder && python -m pytest -q)
node --test tests/position-lock.test.cjs
```

I test alla radice coprono la CLI pubblica da testo a WAV con i modelli effettivi AdaSpeech condizionato da un riferimento e BigVGAN, le funzioni originali dei simboli/del frontend, input di riferimento sintetici e checkpoint di test ridotti inizializzati casualmente. La fixture del predittore acustico della durata è impostata su un output noto, così da verificare indipendentemente le lunghezze attese di token/frame/WAV. Non viene usato alcun modello vocale scaricato o parlato privato.

La copertura include il ritaglio postnet alla lunghezza effettiva e l'ordine CT; testo effettivo e file con più righe attraverso la predizione acustica e il vocoder selezionato; condizionamento con mel di riferimento e forme d'onda; errori per modello/configurazione/riferimento/interprete/dispositivo mancanti; embedding non validi; isolamento dei percorsi/percorsi relativi al chiamante; prevenzione della sovrascrittura di qualsiasi risorsa di input; contratti di sintesi autonoma da mel salvati/WAV; convalida degli array/gestione della sovrascrittura; e una caratterizzazione diretta del frontend per forme d'onda che mostra la differenza di padding attesa.

La suite del vocoder verifica separatamente l'architettura effettiva ridotta con operazioni di I/O reali di checkpoint/WAV/NumPy. Entrambi i casi con Apple MPS esplicito sono stati verificati localmente con il ripiego su CPU disabilitato. La CI Linux usa Python 3.10/3.12; i casi MPS dipendenti dall'hardware vengono saltati in tale ambiente. Il sito conserva tre test di regressione mirati alla riproduzione audio. La nuova pipeline da testo è testata su CPU; l'esecuzione facoltativa dell'encoder pyannote scaricato, la qualità dei checkpoint addestrati e il supporto acustico completo di MPS esulano da questo registro di test.

Un controllo di base del pacchetto wheel installa il progetto in una directory di importazione separata e invoca il comando da console installato dall'esterno del checkout, includendo una sintesi effettiva da testo a WAV. I test dimostrano un flusso dei dati eseguibile e il corretto impacchettamento delle risorse, non la qualità dei checkpoint dell'articolo, l'addestramento/ripresa completi o le metriche dei rilevatori. I risultati esatti delle esecuzioni locali e i checksum dei caricamenti sono conservati nel documento esterno di consegna del manutentore.

Le regressioni per i vecchi checkpoint verificano la lista limitata di scalari NumPy consentiti, il rifiuto di codice pickle arbitrario, la corrispondenza rigorosa dello stato e il ripristino byte per byte dei bin non uniformi di pitch/energia quando le statistiche esterne sono omesse.

Anche i checkpoint completi acustici e V/VR/VRT forniti localmente hanno superato la verifica rigorosa dello stato. Tre controlli end-to-end hanno usato i pesi acustici completi su CPU e V, VR o VRT su Apple MPS con il ripiego su CPU disabilitato, un vettore unitario sintetico di identità a 512 dimensioni e un WAV pubblico di riferimento. Per il testo di prova, lo stesso mel CT predetto [80,50] ha prodotto tre WAV PCM16 mono distinti e non silenziosi, ciascuno con 12800 campioni a 22050 Hz. Ciò verifica la compatibilità eseguibile, non il condizionamento del parlante originale, la qualità del parlato o le metriche dei rilevatori.
