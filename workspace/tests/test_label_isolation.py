import inspect

from src.experiments.attacks.gradient_clustering import cluster_normalized_gradients
from src.split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from src.split_learning.logging.gradient_transcript_logger import (
    ServerGradientTranscriptLogger,
)


def test_server_middle_forward_has_no_label_parameter():
    parameters = inspect.signature(ServerMiddleGModel.forward).parameters
    assert set(parameters) == {"self", "z"}
    assert "y" not in parameters and "labels" not in parameters


def test_server_transcript_api_has_no_label_parameter():
    parameters = inspect.signature(ServerGradientTranscriptLogger.log).parameters
    assert "y" not in parameters
    assert "label" not in parameters
    assert "labels" not in parameters


def test_clustering_api_has_no_true_label_parameter():
    parameters = inspect.signature(cluster_normalized_gradients).parameters
    assert set(parameters) == {"normalized_features", "k", "random_seed"}
    assert not any("label" in name for name in parameters)
