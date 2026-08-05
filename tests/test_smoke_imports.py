import importlib
import pkgutil

import alpha

MODULES = [
    'alpha.config',
    'alpha.data.data_fetcher',
    'alpha.data.data_loader',
    'alpha.features.feature_registry',
    'alpha.mining.preprocessor',
    'alpha.mining.gflownet',
    'alpha.mining.pysr_engine',
    'alpha.mining.fm_regression',
    'alpha.mining.neutralize',
    'alpha.mining.orchestrator',
    'alpha.mining.registry',
    'alpha.mining.safe_ops',
    'alpha.mining.screener',
    'alpha.mining.stats_validator',
    'alpha.mining.walkforward',
    'alpha.mining.audit_logger',
    'alpha.evaluation.backtester',
    'alpha.evaluation.single_factor_test',
    'alpha.evaluation.check_time_stability',
    'alpha.evaluation.compare_benchmark',
    'alpha.evaluation.factor_compare_1',
    'alpha.evaluation.validate_factor_combo',
    'alpha.evaluation.transform_pysr_stablity_check',
    'alpha.analysis.plot_factor_correlation_heatmap',
    'alpha.analysis.calc_jaccard_homogeneity',
    'alpha.analysis.diagnose_single_features',
    'alpha.analysis.decode_gp_formulas',
    'alpha.analysis.p5',
]


def test_all_packages_import():
    for mod in MODULES:
        importlib.import_module(mod)


def test_subpackages_exist():
    subpkg = ['alpha.data', 'alpha.features', 'alpha.mining',
              'alpha.evaluation', 'alpha.analysis', 'alpha.experiments',
              'alpha.experiments.ablations', 'alpha.experiments.baselines',
              'alpha.cli']
    for sp in subpkg:
        importlib.import_module(sp)
