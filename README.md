# KGQApt - Knowledge Graph Question Answering for Portuguese

Abstract:

The search for answers to natural language questions from a Knowledge Graph (KG) is a field known as Knowledge Graph Question Answering (KGQA). This allows users to obtain answers without needing expertise in a specific KG query language like SPARQL. While most existing solutions focus on training Machine Learning (ML) models to convert questions in English into SPARQL queries, few initiatives have been made for languages other than English, such as Portuguese, which is the sixth most spoken language in the world and presents its own linguistic challenges. Unfortunately, these limitations include a small number of datasets to reproduce ML solutions based on English in other languages. Instead of training a complete end-to-end solution, this work presents a modular approach to the task of KGQA in Portuguese, based on five components: (i) Syntax Analyzer, (ii) Question Type Classification, (iii) Concept Mapping, (iv) Query Generation, and (v) Query Ranking. Our contributions include trained models for question classification and query ranking, specifically tailored for the Portuguese language, offering a comprehensive solution for answering questions in natural language from KGs. In addition to a new Relation Linking model for Portuguese, called ptRL. In experiments conducted using the QALD and LCQuAD datasets, the proposed solution achieved an overall F1-score of 41.9\% on QALD and 45.8\% on LCQuAD, outperforming a baseline score of 10.9\%. To the best of our knowledge, this is the first KGQA solution designed for Portuguese that utilizes the standard QALD and LCQuAD datasets.

Follow the instructions bellow to run the system.

1. Preprocess the LC-QuAD dataset, generate 'linked_answer.json' file as the LC-QuAD dataset with golden standard answers
```bat
python lcquad_dataset.py
```
2. Generate the golden answers for LC-QuAD dataset, generate 'lcquad_gold.json' file as LC-QuAD dataset with generated SPARQL queries based on the entities and properties extracted from the correct standard SPARQL query
```bat
python lcquad_answer.py
```
3. Preprocess the LC-QuAD dataset for Tree-LSTM training, split the original Lc-QuAD dataset into 'LCQuad_train.json', 'LCQuad_trial.json', 'LCQuad_test.json' each with 70%\20%\10% of the original dataset. Generate the dependency parsing tree and the corresponding input and output required to train the Tree-LSTM model.
```bat
python learning/treelstm/preprocess_lcquad.py
```
4. Train the Tree-LSTM. The generated checkpoints files are stored in \checkpoints folder and used in lcquad_test.py and qald_test.py
```bat
python learning/treelstm/main.py
```   
5. Generate Phrase Mapping for LC-QuAD test dataset
```bat
python entity_lcquad_test.py
```   
6. Generate Phrase Mapping for QALD-7 test dataset
```bat
python entity_qald.py
```   
7. Test the KGQApt on LC-QuAD test dataset
```bat
python lcquad_test.py
```   
8. Test the KGQApt on LC-QuAD whole
```bat
python lcquadall_test.py
```
9. Test the KGQApt on QALD-7 dataset
```bat
python qald_test.py
```
10. Analyze the Question Type Classification accuracy on LC-QuAD and QALD-7 dataset
```bat
python question_type_anlaysis.py
```
11. Analyze the final result for LC-QuAD and QALD-7 dataset
```bat
python result_analysis.py
```