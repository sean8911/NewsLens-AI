import logging


def configure_logging(level: str) -> None:
    """Keep logs on stderr so stdout remains machine-readable JSON Lines."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
