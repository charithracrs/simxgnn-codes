# SimXGNN: Similarity-Enhanced Cross-layered Graph Neural Network for drug-target interaction

## Requirements

```
python=3.10.10
numpy=1.23.5
torch=2.0.0
scipy=1.10.1
scikit-learn=1.2.2
```

## Run model

The drug and protein embeddings are trained and stored at `data/Trained_embs`. To run our model for drug-target interaction:

- `python train.py`

To learn drug embeddings from drug graph structure data:

- `python train_dti-cnn.py --model_type "pretrain_drug" --epoch "2000" `

To learn protien embeddings from protein graph structure data:

- `python train_dti-cnn.py  --model_type "pretain_protein --epoch "2000" `

## Dataset

### /data

- `drug.txt`: list of drug names
- `protein.txt`: list of protein names
- `disease.txt`: list of disease names
- `se.txt`: list of side effect names
- `drug_dict_map`: a complete ID mapping between drug names and DrugBank ID
- `protein_dict_map`: a complete ID mapping between protein names and UniProt ID
- `mat_drug_se.txt` : Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_protein_drug.txt` : Protein-Drug interaction matrix
- `mat_drug_protein.txt` : Drug_Protein interaction matrix
- `mat_drug_drug.txt` : Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt` : Drug-Disease association matrix
- `Similarity_Matrix_Drugs.txt` : Drug similarity scores based on chemical structures of drugs
- `Similarity_Matrix_Proteins.txt` : Protein similarity scores based on primary sequences of proteins

### /processed

- `drug_smile_structure_edge_list`: contains the edges between the atoms in each drug
- `protein_structure_edge_list` : contains the edges between the amino acid residues in each protein
- `x_list_drug`: atom initial features of all drug
- `x_list_protein`: amino acid residue initial features of all protien

### /similarity

- `sim_mat_drug.txt`: similarity got from Similarity_Matrix_Drugs;Drug similarity scores based on chemical structures of drugs
- `sim_mat_drug_disease.txt`: similarity learned from Drug-Disease association matrix
- `sim_mat_drug_drug.txt`: similarity learned from Drug-Drug interaction matrix
- `sim_mat_drug_protein.txt`: similarity learned from Drug_Protein interaction matrix (transpose of the above matrix)
- `sim_mat_drug_protein_remove_homo.txt`: similarity learned from Drug_Protein interaction matrix, in which homologous proteins with identity score >40% were excluded
- `sim_mat_drug_se.txt`: similarity learned from Drug-SideEffect association matrix
- `sim_mat_protein.txt`: similarity learned from Protein similarity scores based on primary sequences of proteins
- `sim_mat_protein_disease.txt`: similarity learned from Protein-Disease association matrix
- `sim_mat_protein_drug.txt`: similarity learned from Protein-Drug interaction matrix
- `sim_mat_protein_protein.txt`: Protein-Protein interaction matrix
- `association_sim_drug.txt`: combine `sim_mat_drug_drug`, `sim_mat_drug_disease`, `sim_mat_drug_se`get the association similarity
- `association_sim_protein.txt`: combine `sim_mat_protein_disease.txt`, `sim_mat_protein_protein.txt` get the association similarity

### /Trained_embs

- `drug`: drug embs trained from drug graph structure data
- `protein`: protien embs trained from protein graph structure data
