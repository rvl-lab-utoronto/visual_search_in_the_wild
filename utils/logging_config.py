# LOGGING
import logging
""" Logging guide:
* Display ordinary console output: print()
* Status monitoring or fault investigation:  logging.info() (or logging.debug() for detailed output)
* Warning regarding particular runtime event (issue is avoidable): warnings.warn()
* Warning regarding particular runtime event (nothing can be done): logging.warning()
* Report error regarding a particular runtime event: Raise an exception
* Report suppression of error without raising an exception: logging.error(), logging.exception() or logging.critical()
"""
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.ERROR, datefmt='%m/%d/%Y %I:%M:%S %p')
