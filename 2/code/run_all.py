"""Run all homework experiments and save CSV summaries/plots."""
import os
from task1_hddt import run_hddt_experiments
from task2_bagging import run_bagging_experiments
from task3_adaboost import run_adaboost_experiments

if __name__ == '__main__':
    here = os.path.dirname(__file__)
    results_dir = os.path.join(here, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(here, '..', 'figures'), exist_ok=True)
    data_path = os.path.join(here, 'Covid.csv')
    print('Running Task 1...')
    run_hddt_experiments(data_path, save_dir=results_dir)
    print('Running Task 2...')
    run_bagging_experiments(data_path, save_dir=results_dir)
    print('Running Task 3...')
    run_adaboost_experiments(data_path, save_dir=results_dir)
    print('Done. Results in ../results and plots in ../figures')
