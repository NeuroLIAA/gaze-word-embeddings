# Gaze-driven Word Embeddings
![Pipeline](pipeline.png#gh-light-mode-only)
![Pipeline](pipeline_dark.png#gh-dark-mode-only)
## Abstract
Recent work has started exploring the incorporation of eye movement features from reading into the training of specific language tasks, driven by the intuition that reading order, while structured, is not linear. Readers may skip some words, linger on others, or revisit earlier text. This raises the question of how gaze patterns can inform and shape the latent space of language models. In this work, we investigate the integration of gaze information into word embeddings by using text as read by human participants as input and predicting gaze measures from the latent space. To that end, we conducted an eye-tracking experiment with 76 participants reading 20 short stories in Spanish and fine-tuned Word2Vec and AWD-LSTM models on the collected data. We evaluated the resulting embeddings using centered kernel alignment against word embeddings derived from word association tasks and correlations with human similarity judgments of word pairs. While the results were mixed—showing limited gains in some cases and declines in others—this study lays the groundwork for future research. Expanding the linguistic diversity of datasets and employing more cognitively aligned evaluation tasks will be essential to fully uncover the role of gaze information in bridging computational and human language representations.
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
