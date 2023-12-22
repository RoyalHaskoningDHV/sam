from .benchmark import benchmark_wrapper
from .benchmark import (
    benchmark_model,
    plot_score_dicts,
    preprocess_data_for_benchmarking,
)

try:
    from .keras_templates import (
        create_keras_autoencoder_rnn,
        create_keras_autoencoder_mlp,
        create_keras_quantile_mlp,
        create_keras_quantile_rnn,
    )
    from .mlp_model import MLPTimeseriesRegressor

except ImportError:
    pass

from .base_model import BaseTimeseriesRegressor
from .sam_shap_explainer import SamShapExplainer
from .linear_model import LinearQuantileRegression
from .constant_model import ConstantTimeseriesRegressor
from .lasso_model import LassoTimeseriesRegressor


__all__ = [
    "benchmark_wrapper",
    "benchmark_model",
    "plot_score_dicts",
    "preprocess_data_for_benchmarking",
    "create_keras_autoencoder_rnn",
    "create_keras_autoencoder_mlp",
    "create_keras_quantile_mlp",
    "create_keras_quantile_rnn",
    "SamShapExplainer",
    "LinearQuantileRegression",
    "BaseTimeseriesRegressor",
    "ConstantTimeseriesRegressor",
    "LassoTimeseriesRegressor",
    "MLPTimeseriesRegressor",
]
