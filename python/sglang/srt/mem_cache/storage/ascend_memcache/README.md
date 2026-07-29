# Ascend MemCache as L3 KV Cache

This document explains how to use **Ascend MemCache** as the L3 KV Cache backend for **SGLang HiCache**.

Related documentation:

- [Ascend MemCache Build Guide](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md)
- [Ascend MemCache Config Guide](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md)
- [Ascend MemCache Python API](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_python_api.md)
- [SGLang HiCache Design](https://docs.sglang.io/advanced_features/hicache_design.html)
- [Ascend MemFabric](https://gitcode.com/Ascend/memfabric_hybrid)
- [Ascend MemCache](https://gitcode.com/Ascend/memcache)

## About MemCache

MemCache is a distributed cache system from Ascend, built on MemFabric underneath, and can provide a high-performance distributed memory pool.
In SGLang HiCache, MemCache can be used as the L3 KV Cache backend to store and reuse KV cache.

For Kimi K3, one logical cache entry contains both of the following pools:

| SGLang pool | Kimi K3 state | Storage objects |
| --- | --- | --- |
| `kv` | Gated-MLA KV | K and V objects for every logical page |
| `mamba` | KDA recurrent checkpoint | Temporal state and every convolution state |

The entry is usable only when every required object exists. MLA is replicated
across TP ranks, while KDA state is rank-sharded and therefore uses TP/PP/CP
rank-scoped keys. When MLA selects `page_first_kv_split`, SGLang automatically
uses `page_first_direct` for the KDA host sidecar.


## Install Ascend Memcache

[Memcache Official Document](https://gitcode.com/Ascend/memcache/blob/master/doc/install_run.md)

```bash
pip install memcache_hybrid
```

## Deploy MemCache
### Metaservice
add `metaservice_config.json`
```json
{
    "meta_service_url": "tcp://127.0.0.1:5000",
    "config_store_url": "tcp://127.0.0.1:6000",
    "metrics_url": "http://127.0.0.1:8000",
    "log_level": "info",
    "ubs_io_enable": true
}
```

Pass MetaService options via `metaservice_config.json` (see above). Keys below match `memcache_hybrid.MetaConfig` field names.

| Key | Type | Required | Default | Valid range | Description |
| --- | --- | --- | --- | --- | --- |
| `meta_service_url` | string | optional | `tcp://127.0.0.1:5000` | `tcp://<ip>:<port>` | Meta service listen address. Port in [1025, 65535]. |
| `config_store_url` | string | optional | `tcp://127.0.0.1:6000` | `tcp://<ip>:<port>` | Config store address. Port in [1025, 65535]. |
| `metrics_url` | string | optional | `http://127.0.0.1:8000` | `http://<ip>:<port>` | HTTP metrics endpoint. Port in [1025, 65535]. |
| `ha_enable` | boolean | optional | `false` | `true` / `false` | Enable MetaService master/backup HA in a K8s cluster. |
| `log_level` | string | optional | `info` | `debug` / `info` / `warn` / `error` | Log level. |
| `log_path` | string | optional | `/var/log/memcache_hybrid` | relative or absolute path | Log directory. Absolute paths start with `/`. |
| `log_rotation_file_size` | integer | optional | `20` | [1, 500] | Log rotation file size in MB. |
| `log_rotation_file_count` | integer | optional | `50` | [1, 50] | Number of rotated log files to keep. |
| `evict_threshold_high` | integer | optional | `90` | [1, 99] | Eviction high-water mark (%). Max is 99. Eviction is skipped when a single put exceeds 1% of capacity. |
| `evict_threshold_low` | integer | optional | `80` | [0, 98] | Eviction low-water mark (%) after eviction completes. |

For more options, see [MemCache Configuration Guide — MetaService Config](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md#metaservice-config).


## Quick Start Ascend_memcache as L3 backend

### Shell 1: Start Meta service

```bash
python -m sglang.srt.mem_cache.storage.ascend_memcache.start_meta_service --config_path "${metaservice_config_path}"
```

### Shell 2: Start SGLang Server


```bash
python -m sglang.launch_server \
  --model-path ${model_path} \
  --hicache-io-backend kernel_ascend \
  --attention-backend ascend \
  --enable-hierarchical-cache \
  --hicache-storage-backend ascend_memcache \
  --hicache-mem-layout page_first_kv_split \
  --hicache-storage-backend-extra-config '{"meta_service_url":"tcp://127.0.0.1:5000", "config_store_url":"tcp://127.0.0.1:6000", "log_level":"info", "world_size":256, "protocol": "device_sdma", "dram_size": "1GB"}'
```

Pass LocalService options via `--hicache-storage-backend-extra-config` (JSON). Keys below match `memcache_hybrid.LocalConfig` field names.
You can also pass a JSON file as
`--hicache-storage-backend-extra-config @/path/to/localservice_config.json`, or
set `SGLANG_HICACHE_MEMCACHE_CONFIG_PATH` to that file.

| Key | Type | Required | Default | Valid range | Description |
| --- | --- | --- | --- | --- | --- |
| `meta_service_url` | string | optional | `tcp://127.0.0.1:5000` | `tcp://<ip>:<port>` | Meta service address. Port in [1025, 65535]. In HA, `<ip>` is the cluster IP. |
| `config_store_url` | string | optional | `tcp://127.0.0.1:6000` | `tcp://<ip>:<port>` | Config store address. Port in [1025, 65535]. |
| `log_level` | string | optional | `info` | `debug` / `info` / `warn` / `error` | Log level. |
| `world_size` | integer | optional | `256` | [1, 1024] | Max rank count. Cannot change after ranks connect; restart Meta to update. |
| `protocol` | string | **required** | `host_rdma` | `host_rdma`, `host_urma`, `host_tcp`, `host_shm`, `device_sdma`, `device_rdma` | Transport protocol. `host_shm` requires `dram_size` > 0, `hbm_size` = 0, and no hcom. |
| `hcom_url` | string | optional | `tcp://127.0.0.1:7000` | `tcp://<ip>:<port>` | HCOM address for the DRAM pool. Port in [1024, 65535]. |
| `dram_size` | string / integer | **required** | `1GB` | [0, 1TB] | DRAM pool size. Accepts `134217728`, `2048KB`, `200mb`, `2.5G`, `1TB`, etc. Auto-aligned to 2MB (`host_rdma` / `host_tcp` / `host_shm`) or 1GB (`device_sdma` / `device_rdma`). |
| `hbm_size` | string / integer | optional | `0` | [0, 1TB] | HBM pool size (same format as `dram_size`). Must be `0` when using `host_shm`. |
| `max_dram_size` | string / integer | optional | `64GB` | [0, 1TB] | Max `dram_size` across all local processes. |
| `max_hbm_size` | string / integer | optional | `0` | [0, 1TB] | Max `hbm_size` across all local processes. |

## Enable the SSD tier

SSD offload is implemented inside MemCache through UBS_IO. SGLang continues to
use the same batch `put`, `exist`, and `get` calls and does not run a separate
SSD allocator or eviction policy.

Install the UBS_IO KV cache library (`libubsio_kvc.so`) and enable the following
Python field in both the MemCache LocalService and MetaService configuration
files:

```json
{
  "ubs_io_enable": true
}
```

SGLang also accepts the underlying MemCache spelling
`"ock.mmc.ubs_io.enable": true` and maps it to the Python field. If SSD is
requested but the installed `memcache_hybrid` package does not expose UBS_IO
support, startup fails instead of silently falling back to DRAM-only mode.

See the [MemCache UBS_IO usage guide](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_ssd_usage.md)
for installation, SSD mount, and monitoring requirements. This setting is not a
SGLang CLI option.

To verify real SSD fallback, configure a deliberately small `dram_size`, write
more Kimi cache data than that capacity, and confirm all of the following:

1. MemCache metrics report DRAM eviction and SSD occupancy.
2. `batch_is_exist` still reports the complete MLA + KDA entry.
3. `batch_get_into` restores every component and generation matches the
   no-HiCache baseline within the model's normal tolerance.

`--dcp-size > 1` with an L3 storage backend is intentionally rejected until the
storage namespace and MLA ownership rules become DCP-rank aware.


For more options, see [MemCache Configuration Guide — LocalService Config](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md#localservice-config).
