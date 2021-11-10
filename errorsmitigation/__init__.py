# Copyright (c) Microsoft Corporation and ErrorsMitigation contributors.


"""data processing tools to help prepare data for ML training."""

# import atexit
# import logging
# import os

# __name__ = "errorsmitigation"
# __version__ = "0.0.0.1"

# errorsmitigation_logs = os.environ.get('ERRORSMITIGATION_LOGS')

# if errorsmitigation_logs is not None:
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     os.makedirs(os.path.dirname(errorsmitigation_logs), exist_ok=True)
#     handler = logging.FileHandler(errorsmitigation_logs, mode='w')
#     handler.setLevel(logging.INFO)
#     logger.addHandler(handler)
#     logger.info('Initializing logging file for fairlearn')

#     def close_handler():
#         handler.close()
#         logger.removeHandler(handler)

#     atexit.register(close_handler)
