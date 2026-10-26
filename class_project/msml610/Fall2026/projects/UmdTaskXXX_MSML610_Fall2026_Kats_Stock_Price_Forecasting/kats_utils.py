"""
Utility functions for the Kats tutorial notebooks.

The notebooks `kats.API.ipynb` and `kats.example.ipynb` call these functions
and keep only the Kats API calls and the narrative in the cells.

Import as:

import class_project.msml610.Fall2026.projects.UmdTaskXXX_MSML610_Fall2026_Kats_Stock_Price_Forecasting.kats_utils as kats_utils
"""

import logging

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Data loading and preprocessing (filled in Phase 3).
# #############################################################################

# #############################################################################
# Forecasting and baselines (filled in Phase 3).
# #############################################################################

# #############################################################################
# Evaluation and plotting (filled in Phase 3).
# #############################################################################
