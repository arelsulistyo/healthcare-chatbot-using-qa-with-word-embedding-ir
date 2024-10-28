Step 1: Create a virtual environment
```sh
python3 -m venv .venv
source .venv/bin/activate # On Windows, use `.\.venv\Scripts\activate`
```


Step 2: Install requirements

```sh
pip install -r requirements.txt
```

Step 3: Download the datasets

- [MedQuAD dataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- [PubMed 200k RTC](https://www.kaggle.com/datasets/matthewjansen/pubmed-200k-rtc)

Step 4: Download Word2Vec model

- [GoogleNews-vectors-negative300
  ](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)

Your directory should look like this:

```
.
├── GoogleNews-vectors-negative300.bin
├── GoogleNews-vectors-negative300.bin.gz
├── MedQuad-MedicalQnADataset.csv
├── PubMed_200k_RCT
│   ├── dev.csv
│   ├── dev.txt
│   ├── test.csv
│   ├── test.txt
│   ├── train
│   │   └── train.txt
│   └── train.csv
├── PubMed_200k_RCT_numbers_replaced_with_at_sign
│   ├── dev.csv
│   ├── dev.txt
│   ├── test.csv
│   ├── test.txt
│   ├── train
│   │   └── train.txt
│   └── train.csv
├── PubMed_20k_RCT
│   ├── dev.csv
│   ├── dev.txt
│   ├── test.csv
│   ├── test.txt
│   ├── train.csv
│   └── train.txt
├── PubMed_20k_RCT_numbers_replaced_with_at_sign
│   ├── dev.csv
│   ├── dev.txt
│   ├── test.csv
│   ├── test.txt
│   ├── train.csv
│   └── train.txt
├── README.md
├── requirements.txt
└── stbi.ipynb
```

Step 5: Run the notebook
