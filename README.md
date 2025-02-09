# MALGUARD

## MalGuard: Towards Real-Time, Accurate, and Actionable Detection of Malicious Packages in PyPI Ecosystem

### dataset 
 YOU CAN DOWNLOAD THE MALICIOUS PYTHON PACKAGES FROM [(pypi_malregistry)](https://github.com/lxyeternal/pypi_malregistry)


### CODE

First, you need to extract all Python tar.gz archive files to the specified folder. (social-network/dcp_targzfile.py)

Then you can use 'social-network/cal_cen_new.py' to extract social-network graph for every malicious packages.

We provide the feature sets processed by ChatGPT-3.5-turbo, which are as follows:

'social-network/closeness_sensitive_api.json'

'social-network/degree_sensitive_api.json'

'social-network/harmonic_sensitive_api.json'

'social-network/katz_sensitive_api.json'

Then you can use 'train/fea_vec_ex_new.py' to extract feature vector.

and use 'train/ML-trainer-LIME.py' to train ML models




