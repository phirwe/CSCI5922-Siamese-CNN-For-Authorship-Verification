# Data Preprocessing

1. Using `forms.txt` from the [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- Contains information about each sample: author, sentence, etc.
2. Restructure the dataset into top 100 authors
- Keep a threshold (we have considered 1000 px) such that we get a significant amount of data, i.e. lines instead of words.
- We do this because ours is a handwriting recognition problem, the more amount of words in the data the better.
- We have stored this data on a public server [Authors.zip](https://transfer.sh/z8FLg/Authors.zip).
    - Download this along with the dependencies by running `./install_dependencies.sh`
3. Once we get the top 100 authors, we generate the training data using `create_pairs.py`. This divides images randomly into pairs.
- author1 author2 1/0 --> `1 if author1 == author2 else 0`
4. We now have our `train.txt` files according to various sizes.
5. Repeat steps 3 and 4, for validation files: `valid.txt` according to various sizes.


## Now we have our train-test files according to the data preprocessing steps.

