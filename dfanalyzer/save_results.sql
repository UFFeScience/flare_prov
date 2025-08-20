-- FL Job Configuration
COPY (
    SELECT
        trial_id,
        task_name,
        data_path,
        site_num,
        site_name_prefix,
        data_size,
        valid_frac,
        split_method,
        num_rounds
    FROM iJobConfig
)
INTO 'results/iJobConfig.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Training Input Parameters
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        n_clusters,
        batch_size,
        max_iter,
        n_init,
        reassignment_ratio,
        random_state,
        timestamp
    FROM iClientTraining
)
INTO 'results/iClientTraining.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Training Output (local results and timing)
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        center_local,
        count_local,
        center_global,
        training_time,
        timestamp
    FROM oClientTraining
)
INTO 'results/oClientTraining.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Aggregated Global Parameters
COPY (
    SELECT
        trial_id,
        center,
        count,
        assembling_time,
        minibatch_kmeans_time,
        timestamp
    FROM oAssemble
)
INTO 'results/oAssemble.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Validation Metrics
COPY (
    SELECT
        trial_id,
        client_id,
        current_round,
        silhouette_score,
        validation_time,
        timestamp
    FROM oClientValidation
)
INTO 'results/oClientValidation.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';

-- Client Finalization Metadata
COPY (
    SELECT
        trial_id,
        client_id,
        duration,
        timestamp
    FROM oFinalizeClient
)
INTO 'results/oFinalizeClient.csv' ON CLIENT
USING DELIMITERS ',', '\n', '"';