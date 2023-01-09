import os
import re
import scipy.io
import numpy as np


mat_path = '/expanse/projects/nsg/external_users/public/arno/mat_files'
embedding_path = ''

mat_files = os.listdir(mat_path)
embedding_files = os.listdir(embedding_path)

mat_embedding = {}

# make dictionary with mat file(key) embedding (value)
# every 3 segments
for m in mat_files:
    mat_embedding[m] = []
    mat_name = re.search('(\d+)\.', m).group().replace('.', '')
    for e in embedding_files:
        embedding_name = re.search('(\d+)\.', e).group().replace('.', '')
        if mat_name == embedding_name:
            mat_embedding[m].append(e)
            
# make dictionary with mat file(key) embedding (value)
# only first 3 segments
for m in mat_files:
    mat_name = re.search('(\d+)\.', m).group().replace('.', '')
    for e in embedding_files:
        embedding_name = re.search('(\d+)\.', e).group().replace('.', '')
        if mat_name == embedding_name:
            mat_embedding[m] = e
            
            
for m in mat_embedding:
    m_path = os.path.join(mat_path, m)
    mat = scipy.io.loadmat(m_path)
    
    e = mat_embedding[m]
    e_path = os.path.join(embedding_path, embedding)
    embedding = np.load(e)
