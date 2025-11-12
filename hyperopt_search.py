# hyperopt_search.py

import os
import time
import pickle
import logging
from operator import itemgetter

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Import your training utilities (adjust module name if different)
from train import load_and_prepare_data, train_one_run

# Optional: set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
except Exception:
    pass

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------------------------
# Configuration (tune as needed)
# ----------------------------
PROCESSED_DIR = "./preprocessed_events"
SAVE_DIR = "./hyperopt_results"
os.makedirs(SAVE_DIR, exist_ok=True)


MAX_EVALS = 3
LOWFID_EPOCHS = 3
FINAL_EPOCHS = 5
TOP_K = 1

"""MAX_EVALS = 50           # total hyperopt trials (reduce for quick tests)
LOWFID_EPOCHS = 30       # low-fidelity epochs for fast evaluation in objective
FINAL_EPOCHS = 50        # final full training epochs for top-K re-eval
TOP_K = 5                # number of top candidates to re-evaluate at full fidelity"""

# Validation event names (you already selected them)
val_names = [
    '200410062340ANX.npz',
    '200510192044ANX.npz',
    '201103230734ANX.npz',
    '201409161228ANX.npz',
    '201505251428ANX.npz',
    '201901141323ANX.npz',
]

# ----------------------------
# Preload data once
# ----------------------------
logging.info("Loading and preparing data from %s", PROCESSED_DIR)
X_train, Y_train, X_val, Y_val, scalers = load_and_prepare_data(PROCESSED_DIR, val_names)
# Save scalers so test script can use them
np.savez(os.path.join(SAVE_DIR, "scalers.npz"),
         mean_X=scalers['mean_X'], std_X=scalers['std_X'],
         mean_Y=scalers['mean_Y'], std_Y=scalers['std_Y'])
logging.info("Data loaded. Train shape: %s, Val shape: %s", getattr(X_train, "shape", None), getattr(X_val, "shape", None))

# ----------------------------
# Search space (recommended: hp.choice for discrete sets)
# ----------------------------
space = {
    'conv_filters': hp.choice('conv_filters', [8, 16, 32]),
    'conv_kh'     : hp.choice('conv_kh', [5, 9]),
    'conv_kw'     : hp.choice('conv_kw', [3, 5]),
    'lstm_units'  : hp.choice('lstm_units', [64, 128, 256]),
    'dropout'     : hp.uniform('dropout', 0.0, 0.5),
    'l2_reg'      : hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-3)),
    'lr'          : hp.loguniform('lr', np.log(1e-5), np.log(1e-3)),
    'batch_size'  : hp.choice('batch_size', [8, 16, 32]),
}

# ----------------------------
# Objective
# ----------------------------
def _normalize_and_cast_params(raw_params):
    """
    Defensive conversion to Python native types and ints where needed.
    raw_params likely contains values directly (because we used hp.choice).
    """
    p = {}
    # ensure keys exist and cast
    p['conv_filters'] = int(raw_params.get('conv_filters'))
    p['conv_kh'] = int(raw_params.get('conv_kh'))
    p['conv_kw'] = int(raw_params.get('conv_kw'))
    p['lstm_units'] = int(raw_params.get('lstm_units'))
    p['dropout'] = float(raw_params.get('dropout'))
    p['l2_reg'] = float(raw_params.get('l2_reg'))
    p['lr'] = float(raw_params.get('lr'))
    p['batch_size'] = int(raw_params.get('batch_size'))
    return p

def objective(raw_params):
    """
    raw_params: dictionary provided by hyperopt (may contain actual values or indices depending on space).
    We will cast/massage them and call train_one_run.
    """
    start_t = time.time()
    # Map and cast params to exact Python types
    params = _normalize_and_cast_params(raw_params)

    logging.info("Trial start - params: %s", params)
    tmp_ckpt = os.path.join(SAVE_DIR, f"tmp_trial_{int(time.time()*1000)}.h5")

    try:
        # Run a low-fidelity training (fast evaluation). Use verbose=0 to reduce clutter.
        val_rmse, model, history = train_one_run(
            params,
            X_train, Y_train, X_val, Y_val,
            epochs=LOWFID_EPOCHS,
            batch_size=params['batch_size'],
            ckpt_path=tmp_ckpt,
            verbose=0
        )
    except Exception as e:
        logging.exception("Trial failed with exception")
        # A failing trial should return a large loss so hyperopt avoids it
        return {'loss': float("inf"), 'status': STATUS_OK, 'params': params, 'exception': str(e)}

    elapsed = time.time() - start_t
    logging.info("Trial finished in %.1f s - val_rmse: %.6f", elapsed, float(val_rmse))

    # cleanup temporary checkpoint to avoid disk accumulation (optional)
    try:
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
    except Exception:
        pass

    # return result with params for later analysis
    return {'loss': float(val_rmse), 'status': STATUS_OK, 'params': params}

# ----------------------------
# Run hyperopt TPE search
# ----------------------------
trials = Trials()
logging.info("Starting hyperopt fmin: max_evals=%d", MAX_EVALS)
rng = np.random.default_rng(RANDOM_SEED)
best_raw = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=trials, rstate=rng)

# Map raw best -> actual param values (space_eval handles hp.choice mappings)
try:
    best_params = space_eval(space, best_raw)
except Exception:
    # if space_eval fails for any reason, try reading params from trials best entry
    logging.warning("space_eval failed, extracting best params from trials")
    sorted_by_loss = sorted([t['result'] for t in trials.trials if 'result' in t and t['result'].get('params') is not None],
                            key=itemgetter('loss'))
    best_params = sorted_by_loss[0]['params'] if sorted_by_loss else None

logging.info("Hyperopt finished. best_params (raw mapped): %s", best_params)

# Save trials and best
with open(os.path.join(SAVE_DIR, "trials.pkl"), "wb") as f:
    pickle.dump(trials, f)
with open(os.path.join(SAVE_DIR, "best_raw.pkl"), "wb") as f:
    pickle.dump(best_raw, f)
with open(os.path.join(SAVE_DIR, "best_params.pkl"), "wb") as f:
    pickle.dump(best_params, f)

# ----------------------------
# Top-K re-evaluation (full-fidelity)
# ----------------------------
logging.info("Collecting top-K candidates from trials for full-fidelity re-eval (K=%d)", TOP_K)
# Extract results that have params and finite loss
valid_results = []
for t in trials.trials:
    r = t.get('result', {})
    loss = r.get('loss')
    params = r.get('params')
    if params is not None and loss is not None and np.isfinite(loss):
        valid_results.append((float(loss), params))

if not valid_results:
    logging.error("No valid trials found to re-evaluate. Exiting.")
    raise SystemExit(1)

valid_results_sorted = sorted(valid_results, key=lambda x: x[0])
top_candidates = valid_results_sorted[:TOP_K]

best_overall = {'val_rmse': float("inf"), 'params': None, 'ckpt': None}

for idx, (loss_val, params) in enumerate(top_candidates):
    logging.info("Re-evaluating top candidate %d/%d: loss=%.6f params=%s", idx+1, len(top_candidates), loss_val, params)
    final_ckpt = os.path.join(SAVE_DIR, f"final_topk_{idx}.h5")
    try:
        val_rmse, model, history = train_one_run(
            params,
            X_train, Y_train, X_val, Y_val,
            epochs=FINAL_EPOCHS,
            batch_size=params['batch_size'],
            ckpt_path=final_ckpt,
            verbose=1
        )
    except Exception as e:
        logging.exception("Full-fidelity re-eval failed for candidate %s", params)
        continue

    logging.info("Candidate %d full-fidelity val_rmse=%.6f (was %.6f)", idx+1, val_rmse, loss_val)

    if val_rmse < best_overall['val_rmse']:
        best_overall['val_rmse'] = float(val_rmse)
        best_overall['params'] = params
        best_overall['ckpt'] = final_ckpt

# Save the best_final model weights as best_model.h5 for test script to load
if best_overall['ckpt'] is not None and os.path.exists(best_overall['ckpt']):
    final_dest = "best_model.h5"
    # copy the best ckpt to expected filename (weights only)
    try:
        import shutil
        shutil.copyfile(best_overall['ckpt'], final_dest)
        logging.info("Saved best model weights to %s (val_rmse=%.6f)", final_dest, best_overall['val_rmse'])
    except Exception:
        logging.exception("Failed to copy final ckpt to %s", final_dest)
else:
    logging.warning("No successful final candidate to save as best_model.h5")

# Save summary.json-like info
summary = {
    'best_hyperopt_mapped': best_params,
    'best_full_reval': best_overall,
    'max_evals': MAX_EVALS,
    'lowfid_epochs': LOWFID_EPOCHS,
    'final_epochs': FINAL_EPOCHS,
    'timestamp': time.time(),
}
with open(os.path.join(SAVE_DIR, "search_summary.pkl"), "wb") as f:
    pickle.dump(summary, f)

logging.info("Hyperparameter search completed. Summary saved to %s", SAVE_DIR)
