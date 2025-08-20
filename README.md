# Federated k-Means Clustering with Scikit-learn

## Overview
This project demonstrates Federated Learning (FL) with k-Means clustering using Scikit-learn. The aggregation follows the MiniBatch k-Means approach, where clients perform local training, and a central server aggregates their results to update global cluster centers.

### Federated k-Means Process
Each FL round consists of:
1. **Local Training**: Clients initialize with global centers and train MiniBatchKMeans on local data.
2. **Global Aggregation**: The server collects cluster centers and counts from all clients, updates the global model, and redistributes it for the next round.

#### Initialization Strategy
- Clients use k-means++ for initial cluster centers.
- The server aggregates initial centers using a round of k-means to determine the global starting point.

---

## Running Federated Learning with Proper Data Splitting

### Data Partitioning
For an trial with `K` clients, the dataset is split into `K+1` non-overlapping parts:
- `K` parts for client training.
- `1` common validation dataset.

### Simulating Data Imbalance
Clients' data sizes can be distributed using different strategies:
- **Uniform**: Equal data for all clients.
- **Linear**: Data size grows linearly with client ID.
- **Square**: Data size increases quadratically.
- **Exponential**: Best suited for small `K` (e.g., `K=5`) to simulate extreme imbalance.

---

## Capturing Provenance with DfAnalyzer

DfAnalyzer is a library designed to capture provenance data, which includes:
- **Prospective Provenance**: The "recipe" of the trial, capturing configurations before execution.
- **Retrospective Provenance**: Logs and tracks execution results, including data transformations and FL training steps.

### Setting Up DfAnalyzer
1. Load the DfAnalyzer Docker image:
   ```bash
      docker pull nymeria0042/dfanalyzer   
   ```
2. Deploy the DfAnalyzer container:
   ```bash
   cd dfanalyzer && docker compose up dfanalyzer
   ```
3. Ensure DfAnalyzer is running in the background before starting trials.

4. Create and activate a virtualenv with python=3.8
   ```bash
   virtualenv venv --python=3.8
   . venv/bin/activate
   pip install -r requirements.txt
   ```

5.  Install `dfa-lib-python`:
   ```bash
   cd dfanalyzer/dfa-lib/python & make install
   ```
6. Run the prospective provenance script:
   ```bash
   python fed-clustering/utils/prospective_provenance.py
   ```
   
   - Responsible for capturing and recording metadata about the design and configuration of the trials before the runs.

---

## Federated Learning with NVFlare
NVFlare is used to set up the federated learning infrastructure.

### Setting up NVFLARE
- Build the NVFlare image
```bash
docker build -t nvflare-service .
```



## Running the trial
From `fed-clustering folder`.

### Prepare trial
```bash
source start_trial.sh
```

- If versioning_control is enabled in `utils/start_trial.py`, a new branch is created under the `trials/` folder, named with the user and the trial’s start timestamp. A hash is generated and stored in `trial_info.json`. Additionally, a commit is created containing this hash. The hash can be used to query the provenance database for records related exclusively to that trial.

- If versioning_control is disabled, the trial runs in the current folder and branch without creating a new branch or commit. Nonetheless, a hash is still generated and stored in `trial_info.json`, enabling tracking and querying in the provenance database.

- Remember to reactivate the virtual environment

### Preparing Data and Configuration
```bash
source prepare_data.sh
source prepare_job_config.sh
```

### Provisioning
- Run:
```bash
nvflare provision
```
This creates a `workspace/fed_clustering` directory with the following structure:
```
workspace
└── fed_clustering
    ├── prod_01
    │   ├── admin@nvidia.com
    │   ├── server1
    │   ├── site-1
    │   ├── site-2
    │   ├── overseer
    │   └── compose.yaml
    └── resources
```
- Manually copy `workspace/fed_clustering/prod_00/.env` and `workspace/fed_clustering/prod_00/compoose.yaml` to the new `workspace/fed_clustering/prod_01/` folder 

---
### Running NVFlare
- Navigate to `prod_01` and launch FL components:
```bash
docker compose up
```

- Manually copy `jobs/sklearn_2_uniform` to `workspace/fed_clustering/prod_01/admin@nvidia.com/transfer/`.

### Distributing Data to Clients
- Create dataset folders inside the containers:
```bash
docker exec -it site-1 mkdir -p /tmp/nvflare/dataset
```
- Copy data to each client:
```bash
docker cp /tmp/nvflare/dataset/des.csv site-1:/tmp/nvflare/dataset
```
- Repeat for all sites.


### Configuring the Hosts File
1. Get the local hostname:
   ```bash
   hostname -I | awk '{print $1}'
   ```
2. Add it to `/etc/hosts`:
   ```bash
   sudo vim /etc/hosts
   ```
   Add:
   ```
   {IP} server1 overseer
   ```

### Running the FL Server
Inside `prod_01`, start the FL admin panel:
```bash
./admin@nvidia.com/startup/fl_admin.sh
```
Log in with:
```
admin@nvidia.com
```

#### Checking FL System Status
```bash
check_status [server|client]
```

### Submitting and Running the trial
```bash
submit_job sklearn_kmeans_2_uniform
```

### Monitor DfAnalyzer for provenance tracking.
#### Query the provenance database
```bash 
docker exec -it dfanalyzer mclient -u monetdb -d dataflow_analyzer
```

The default password is `monetdb`.

Then, we can submit the queries, like:

```SQL
SELECT client_id, silhouette_score FROM iClientValidation WHERE trial_id = {hash_trial};
```
---

## Saving the retrospective provenance data using the `save_results` script
```bash 
docker exec -it dfanalyzer mclient -u monetdb -d dataflow_analyzer -i save_results.sql
```

This creates a folder `results` inside the `dfanalyzer` directory with .csv files for each predefined provenance table 

## Conclusion
This project demonstrates federated k-Means clustering using NVFlare, Scikit-learn, and DfAnalyzer. Provenance data is captured throughout, ensuring transparency and reproducibility of FL trials.

