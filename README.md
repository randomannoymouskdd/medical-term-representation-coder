# Coder: Cross-Lingual Medical Term Representation via Contrastive Learning on Knowledge Graph
Codes and models.

# Models download
Download from [Google Drive](https://drive.google.com/file/d/13b63wVNsXAwLMHK3MfwNROhC72Yxo9Lu/view?usp=sharing).

# Get UMLS2020AA
We use UMLS2020AA as our training sources.
Download from [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2020AA).

# Train your model
```shell
# Prepare UMLS Dataset
mkdir umls
# Download MRCONSO.RRF, MRREL.RRF and MRSTY.RRF to umls floder

cd pretrain
# For training English Version
sh start_eng.sh
# For training cross-lingual Version
sh start_all.sh
```

# Test
## CADEC & PsyTar
Embedding-based term normalization are tested on CADEC and PsyTar.
Only use testing datasets here.
```shell
cd test
python cadec/cadec_eval.py bert_model_name_or_path
python cadec/cadec_eval.py word_embedding_path
```

## MCSM
You can modify bert_model_name_or_path in **test/embeddings_reimplement/mcsm.py**.
```shell
cd test/embeddings_reimplement
python mcsm.py
```
- **mcsm.py** supports bert, word, cui type embeddings.

## DDBRC
Only sample datas are provided to show data format.
Full training datas are not shown according to licences of Diseases Database. 
Please contact [Diseases Database](http://www.diseasesdatabase.com/) for full data. 
```shell
cd test/diseasedb
python train.py your_embedding embedding_type freeze_or_not gpu_id
```
- your_embedding should contain an embedding or a pre-trained model
- embedding_type should be in [bert, word, cui]
- freeze_or_not should be in [T, F], T means freeze the embedding, and F means fine-tune the embedding
- gpu_id should be a number, e.g. 0

## mantra
Download [the Mantra GSC](https://files.ifi.uzh.ch/cl/mantra/gsc/GSC-v1.1.zip) and unzip the xml files to /test/mantra/dataset, run
```shell
cd test/mantra
python test.py
```
