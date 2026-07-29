import argparse
import json
import logging
import os
import sys
from typing import Any

from memcache_hybrid import MetaConfig, MetaService

logger = logging.getLogger("ascend_memcache.start_meta_service")

_META_CONFIG_ALIASES = {
    "ock.mmc.ubs_io.enable": "ubs_io_enable",
}


def _load_json_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a JSON object: {config_path}")
    return data


def _apply_meta_config(config: MetaConfig, data: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    for key, value in data.items():
        field = _META_CONFIG_ALIASES.get(key, key)
        if hasattr(config, field):
            setattr(config, field, value)
        else:
            unknown.append(field)
    return unknown


def launch_meta_service(config_path: str) -> int:
    try:
        config_data = _load_json_config(config_path)
    except Exception as e:  # noqa: BLE001 - CLI boundary reports configuration errors
        logger.error("Failed to load meta service config from %s: %s", config_path, e)
        return 1

    meta_cfg = MetaConfig()
    unknown = _apply_meta_config(meta_cfg, config_data)
    requested_ubs_io = bool(
        config_data.get("ubs_io_enable") or config_data.get("ock.mmc.ubs_io.enable")
    )
    if requested_ubs_io and "ubs_io_enable" in unknown:
        logger.error(
            "UBS_IO SSD was requested, but this memcache_hybrid version does not "
            "expose MetaConfig.ubs_io_enable"
        )
        return 1
    if unknown:
        logger.warning("Ignoring unknown MetaConfig keys: %s", unknown)

    try:
        setup_ret = MetaService.setup(meta_cfg)
        if isinstance(setup_ret, int) and setup_ret != 0:
            logger.error("MetaService.setup failed, ret=%s", setup_ret)
            return setup_ret
        logger.info(
            "MetaService setup succeeded with config=%s, ubs_io=%s",
            config_path,
            bool(getattr(meta_cfg, "ubs_io_enable", False)),
        )
        MetaService.main()
        return 0
    except KeyboardInterrupt:
        logger.info("MetaService interrupted by user.")
        return 0
    except Exception as e:  # noqa: BLE001 - CLI boundary reports backend failures
        logger.error("MetaService failed to run: %s", e)
        return 2


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "metaservice_config.json")

    parser = argparse.ArgumentParser(
        description="Launch Ascend MemCache MetaService via JSON."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=default_path,
        help=f"Path to meta service JSON config (default: {default_path})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return launch_meta_service(args.config_path)


if __name__ == "__main__":
    sys.exit(main())
