# Addestramento

Il progetto include implementazioni di addestramento acustico e del vocoder. Le definizioni dei modelli, i metodi forward di addestramento, i calcoli delle perdite e le chiavi dei parametri dei checkpoint sono conservati. Le modifiche al percorso di inferenza supportato non dimostrano che l'esperimento di addestramento completo dell'articolo originale sia stato recuperato.

## Modello acustico

Il punto di ingresso per l'addestramento acustico è `acoustic/train_en_cn.py`, con i file YAML del modello/preprocessing/addestramento sotto `acoustic/config/`. Il fine-tuning è in `acoustic/finetune.py`. Questi programmi richiedono un corpus esterno preelaborato contenente mel, pitch, energia, durate, aggregati dei fonemi, metadati/embedding dei parlanti e statistiche corrispondenti. Ottieni l'ambiente originale associato al checkpoint e i manifest prima di eseguire/riprendere l'esperimento.

Esegui l'addestramento nel proprio ambiente e nella directory del componente; il gruppo di dipendenze per l'inferenza non dichiara una copertura completa di preprocessing/addestramento. `acoustic/requirements.txt` è un registro storico delle dipendenze con vincoli di versione obsoleti e in conflitto, non la procedura supportata per l'inferenza. Esamina le sue dipendenze di addestramento e i percorsi di configurazione rispetto all'esperimento previsto, invece di installarlo sopra un ambiente di inferenza.

## Vocoder e obiettivi avversari

Il punto di ingresso per l'addestramento del vocoder è `vocoder/train.py`. V indica la sintesi di base, VR prende di mira RawNet2 e VRT prende di mira RawNet2/TSSD. Nel metodo previsto, AASIST è escluso dal fine-tuning del vocoder. L'attuale programma sperimentale di addestramento contiene percorsi dei rilevatori inattivi/commentati e differenze irrisolte relative a gradienti/etichette/ottimizzatore/perdite/decadimento. Inoltre importa un file `audio_utils.py` mancante attraverso le utilità RawNet.

Per risolvere tali differenze sono necessari gli script originali di addestramento V/VR/VRT, i pesi/configurazioni dei rilevatori, l'ordine delle classi, i registri delle suddivisioni dei dati/del preprocessing e i log. Non considerare il ripristino di codice commentato o la modifica delle perdite una ricostruzione verificata. Il comportamento dell'addestramento resta invariato finché non sarà disponibile il bundle dell'esperimento originale.

## Ripresa e valutazione

Usa combinazioni esatte di modello/configurazione/checkpoint. La ripresa dell'addestramento acustico può richiedere lo stato dell'ottimizzatore oltre a `model`. La ripresa del vocoder richiede lo stato del discriminatore e dell'ottimizzatore, epoca/step e generatore. Convalida un piccolo percorso di addestramento/ripresa/valutazione sull'hardware originariamente supportato prima di un'esecuzione completa. Conferma la polarità dei punteggi e definizioni coerenti di AUC/EER/accuratezza bilanciata prima del confronto con il manoscritto. I test di inferenza non attestano alcuna esecuzione di addestramento o risultato qualitativo pari a quello dell'articolo.
