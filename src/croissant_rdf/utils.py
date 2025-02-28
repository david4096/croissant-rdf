import logging

# Disable logger in your code with:
# logging.getLogger("croissant_rdf").setLevel(logging.WARNING)
logger = logging.getLogger("croissant_rdf")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
