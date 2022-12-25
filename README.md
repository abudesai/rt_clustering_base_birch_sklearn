BIRCH Model in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- BIRCH
- clustering
- sklearn
- python
- pandas
- numpy
- flask
- nginx
- uvicorn
- docker

This is a Clustering Model that uses BIRCH implemented through Sklearn.

Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) is a clustering algorithm that can cluster large datasets by first generating a small and compact summary of the large dataset that retains as much information as possible. This smaller summary is then clustered instead of clustering the larger dataset

The data preprocessing step includes:

- for numerical variables
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, Sklearn, and feature-engine are used for the data preprocessing steps.

The model includes an inference service with 2 endpoints: /ping for health check and /infer for predictions of nearest clusters in real time. The inference service is implemented using flask+nginx+uvicorn.
