# Gaze-driven Word Embeddings
![Pipeline](pipeline.png#gh-light-mode-only)
![Pipeline](pipeline_dark.png#gh-dark-mode-only)
## Abstract
Reading, while structured, is a non-linear process. Readers may skip some words, linger on others, or revisit earlier text. Emerging work has started exploring the incorporation of reading behaviour through eye-tracking into the training of specific language tasks. In this work, we investigate the broader question of how gaze data can shape word embeddings by using text as read by human participants and predicting gaze measures from them. To that end, we conducted an eye-tracking experiment with 76 participants reading 20 short stories in Spanish and fine-tuned Word2Vec and LSTM models on the collected data. Evaluations with representational similarity analysis and word pair similarities showed a limited, but largely consistent, gain from gaze incorporation, suggesting future work should expand linguistic diversity and use cognitively aligned evaluations to better understand its role in bridging computational and human language representations.
## How to run
### Pretraining
To pretrain the models on Wikipedia, run the following command:
```bash
python train.py <name> --corpora all_wikis --source remote --model <model>
```
Where ```<name>``` is the name of the parent folder in which the trained models will be saved. ```<model>``` is either ```skip``` or ```lstm```. In the case of ```lstm```, set ```--lr``` to ```30```. 
### Fine-tuning
To fine-tune the models, run the following command:
```bash
python train.py <name> --corpora <corpus> --source local --model <model> --finetune <model_path>
```
Where ```<name>``` is the name of the parent folder in which the pretrained models were saved. ```<corpus>``` is either ```texts``` or ```scanpaths``` and ```<model>``` is either ```skip``` or ```lstm```. ```<model_path>``` is the path (relative to ```<name>```) to the pretrained baseline model.

To include gaze measure prediction, add ````--gaze_features```` followed by the gaze measures to predict (e.g. ````--gaze_features ffd fprt tfd````).
### Evaluation
To evaluate the models, run the following command:
```bash
python test.py all_wikis --words_similarities <word_pairs_file>
```
Where ```<word_pairs_file>``` is either ```evaluation/simlex.csv```, ```evaluation/abstract.csv```, or ```evaluation/concrete.csv```.
### Dependencies
This code was tested on Python 3.10 and greater. To install the required dependencies, run:
```bash
pip install -r requirements.txt
```
## How to cite us
```
@inproceedings{travi-etal-2025-exploring,
    title = "Exploring the Integration of Eye Movement Data on Word Embeddings",
    author = "Travi, Ferm{\'i}n  and
      Leclercq, Gabriel Aim{\'e}  and
      Slezak, Diego Fernandez  and
      Bianchi, Bruno  and
      Kamienkowski, Juan E",
    editor = "Kuribayashi, Tatsuki  and
      Rambelli, Giulia  and
      Takmaz, Ece  and
      Wicke, Philipp  and
      Li, Jixing  and
      Oh, Byung-Doh",
    booktitle = "Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.cmcl-1.9/",
    pages = "55--65",
    ISBN = "979-8-89176-227-5"
}
```
