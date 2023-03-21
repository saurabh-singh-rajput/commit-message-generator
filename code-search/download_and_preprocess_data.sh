cd data
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
python preprocess.py
rm -r python
rm -r *.pkl
rm python.zip
cd ..