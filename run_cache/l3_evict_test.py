#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 cache LOAD(prefetch) 验证脚本

流程:
  1) 请求A : 共享前缀 P(固定, 约 57.6K tokens) + 问题 Q1
             -> 触发 L1 写入 -> write_through 写 L2(host) -> 异步写 L3(远端存储)
             -> 必须等 prefill 日志出现 [L3-BACKUP] 后再进入下一步!
  2) N 个互不相同的干扰长请求
             -> 把 device(L1) 和 host(L2) 池都占满, 彻底淘汰共享前缀 P 的节点
  3) 请求B : 相同的共享前缀 P + 问题 Q2
             -> L1 miss + L2 miss -> 期望触发 L3 prefetch

判断标准(prefill 节点日志, 按顺序出现):
  - 步骤1 后:   [L3-BACKUP] Backed up N pages to storage ...   (L3 写入成功)
  - 请求B 时:   HiCache prefetch_from_storage: ISSUED req_id=...   (prefetch 发起)
                [L3-HIT] Prefetching N pages (M tokens) for request ...   (L3 命中)
                HiCache prefetch success ... loaded=N>0   (真正从 L3 读回 N tokens)
  - 反向对照:   换一个从未写过的前缀应看到 [L3-MISS] / prefetch_from_storage: SKIP

重要: 干扰总量必须超过 L1+L2 的池子总容量, 否则 P 留在 L2,
      请求B 会走 [L2-LOAD] 而不是 L3 prefetch。
      池子容量可从 prefill 日志的 "full token usage" 估算:
        近似 L1 容量(token) = 请求token数 / full_token_usage
        近似 L2 容量       = L1 容量 x --hicache-ratio (本例 2.0)
      干扰总量 = 干扰数 x INTERFERE_TOKENS 应显著大于 L1+L2 容量。
      若请求B 仍走 L2(出现 [L2-LOAD]), 继续加大干扰数或调小
      --hicache-ratio / --mem-fraction-static。

decode offload 补充(可选, 独立路径):
  decode 节点 d.sh 开了 --disaggregation-decode-enable-offload-kvcache,
  其日志关键词: Enable offload kv cache for decode side / Finished backup request ...

用法:
  python3 l3_evict_test.py [server] [干扰请求数] [干扰文本长度]
  例: python3 l3_evict_test.py http://127.0.0.1:9903 30 100000

注意: 干扰请求的 KV 也会写入 L3, 干扰总量过大(> L3 池容量)会把
      共享前缀 P 从 L3 里挤掉, 导致请求B 在 L3 也 miss(全重算)。
      干扰总量应刚好超过 L1+L2 容量即可, 不要过量。
"""
import random
import sys
import time

import requests

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9903"
NUM_INTERFERENCE = int(sys.argv[2]) if len(sys.argv) > 2 else 30
INTERFERE_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
GENERATE_URL = f"{SERVER}/generate"

# 共享前缀长度(token 数), 与 perf.sh 的 --gsp-system-prompt-len 一致
PREFIX_TOKENS = 57_600
# 步骤1 后等待 L3 异步写入完成的时间(s)。L3 写入走后台 backup 线程,
# 57.6K tokens 的 backup 需要时间, 不够会出现步骤3 prefetch 无数据
L3_BACKUP_WAIT = 30

# ---------------- 文本生成 ----------------
_SENTENCES = [
    "在工业生产中,原材料的采购成本往往占据总成本的一半以上,因此供应链管理显得尤为重要。",
    "机器学习模型在训练过程中需要大量的高质量数据,数据的质量直接决定了模型的上限。",
    "分布式系统的设计需要权衡一致性与可用性,这是著名的CAP定理所阐述的核心思想。",
    "自然语言处理任务中,词向量的表示方式经历了从静态词向量到上下文相关的演化。",
    "高性能计算集群通常采用InfiniBand网络来连接计算节点,以获得更低的通信延迟。",
    "编译器优化的目标是在不改变程序语义的前提下,尽可能提高目标代码的执行效率。",
    "数据库事务的隔离级别决定了并发事务之间相互影响的程度,分为四个标准级别。",
    "深度学习框架的自动求导机制通过计算图的反向传播来高效地计算参数的梯度。",
    "操作系统中的虚拟内存技术允许程序使用比物理内存更大的地址空间来运行。",
    "软件测试中的边界值分析是一种有效的用例设计方法,能够发现大量隐藏的缺陷。",
    "计算机网络中的TCP协议通过三次握手建立连接,并通过滑动窗口进行流量控制。",
    "嵌入式系统开发通常需要考虑功耗、实时性和资源受限等特殊约束条件。",
    "加密算法分为对称加密和非对称加密两大类,各自适用于不同的应用场景。",
    "云计算平台通过虚拟化技术将物理资源抽象为可按需分配的逻辑资源池。",
    "图神经网络通过消息传递机制在节点之间交换特征信息,从而学习图结构数据。",
    "强化学习中的智能体通过与环境的交互来学习最优策略,以最大化累计回报。",
    "大数据处理框架采用分而治之的思想,将大规模计算任务分解为多个子任务并行执行。",
    "容器化技术的核心是利用命名空间和控制组来实现资源的隔离与限制。",
    "搜索引擎的排序算法综合考虑了文本相关性、网页权威性和用户行为等多种因素。",
    "微服务架构将单体应用拆分为多个独立部署的小服务,提高了系统的可维护性。",
    "量子计算利用量子叠加和量子纠缠的特性,在特定问题上具有经典计算无法比拟的优势。",
    "网络安全中的入侵检测系统通过分析网络流量和系统日志来发现潜在的攻击行为。",
    "编译器前端负责词法分析、语法分析和语义分析,将源代码转换为中间表示。",
    "进程调度算法决定了多个就绪进程在CPU上的执行顺序,影响系统的响应时间。",
    "图像处理中的卷积操作通过滑动窗口对像素邻域进行加权求和,实现特征提取。",
    "区块链技术通过分布式账本和共识机制,在不信任的参与者之间建立信任。",
    "软件架构中的设计模式提供了解决常见问题的可复用方案,降低了系统设计的复杂度。",
    "缓存技术的核心思想是利用局部性原理,将频繁访问的数据存储在更快的存储介质上。",
    "并行编程模型包括共享内存模型和消息传递模型,适用于不同的硬件架构。",
    "机器学习中的正则化技术通过在损失函数中加入惩罚项来防止模型过拟合。",
    "内存管理中的页式存储将逻辑地址空间划分为固定大小的页,便于内存的分配与管理。",
    "负载均衡算法将请求分发到多个后端服务器,以平衡各服务器的负载压力。",
]


def make_text(n_tokens: int, seed: int) -> str:
    """按近似 token 数生成一段确定性的长文本 (1 个中文字符约 1 个 token)。"""
    rng = random.Random(seed)
    # 留 20% 余量, 保证文本长度足够
    n_chars = int(n_tokens * 1.2)
    parts = []
    total = 0
    while total < n_chars:
        s = rng.choice(_SENTENCES)
        parts.append(s)
        total += len(s)
    return "".join(parts)


def make_interfere_text(seed: int) -> str:
    """生成一条与其他干扰请求互不相同的长文本。"""
    return make_text(INTERFERE_TOKENS, seed)


# 共享前缀 P: 固定 seed, 保证请求A/请求B 的 P 完全一致
SHARED_PREFIX = make_text(PREFIX_TOKENS, seed=20260804)
QUESTION_A = "根据以上说明,请回答第一个具体问题: 请简要总结文中的核心观点,并给出你的理由。"
QUESTION_B = "根据以上说明,请回答第二个具体问题: 请分析文中所述方法的优缺点,并给出改进建议。"

SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 32}


def send(text: str, tag: str):
    t0 = time.time()
    resp = requests.post(
        GENERATE_URL,
        json={"text": text, "sampling_params": SAMPLING_PARAMS},
        timeout=1800,
    )
    dt = time.time() - t0
    status = "OK" if resp.status_code == 200 else f"HTTP {resp.status_code}"
    print(f"[{tag}] {status} 耗时 {dt:.1f}s 文本长度 {len(text)} 字符")
    if resp.status_code != 200:
        print(f"[{tag}] body: {resp.text[:500]}")


def main():
    print(f"server={SERVER} 干扰请求数={NUM_INTERFERENCE}")
    print(f"共享前缀约 {PREFIX_TOKENS} tokens, 干扰每条约 {INTERFERE_TOKENS} tokens\n")

    # STEP 1: 请求A (写 L1 -> L2 -> L3)
    print("=== STEP 1: 请求A (共享前缀P + Q1), 写入 L1/L2/L3 ===")
    send(SHARED_PREFIX + QUESTION_A, "请求A")

    # L3 写入是异步的(backup 线程)。等 L3-BACKUP 日志出现再继续。
    print(f"\n等待 {L3_BACKUP_WAIT}s 让 L3 异步写入完成...")
    print(f"请到 prefill 节点日志确认出现: [L3-BACKUP] Backed up N pages to storage")
    time.sleep(L3_BACKUP_WAIT)
    print("等待结束, 若日志中没有 [L3-BACKUP], 请先排查 L3 配置后再继续。\n")

    # STEP 2: 干扰请求, 同时挤掉 P 的 device(L1) 和 host(L2) 节点
    print(f"=== STEP 2: {NUM_INTERFERENCE} 条互不相同的干扰请求 (挤占 L1 + L2 池) ===")
    for i in range(NUM_INTERFERENCE):
        send(make_interfere_text(seed=1000 + i), f"干扰{i + 1}/{NUM_INTERFERENCE}")
        time.sleep(1)

    # STEP 3: 请求B (同样的 P + Q2), 期望 L3 prefetch
    print("=== STEP 3: 请求B (同样的共享前缀P + Q2), 期望 L3 prefetch ===")
    send(SHARED_PREFIX + QUESTION_B, "请求B")

    print("\n请到 prefill 节点日志确认(按顺序):")
    print("  1) 请求B 前出现: HiCache prefetch_from_storage: ISSUED req_id=...")
    print("  2) [L3-HIT] Prefetching N pages (M tokens) for request ...")
    print("  3) HiCache prefetch success ... loaded=N>0   (N 为真正从 L3 读回的 token 数)")
    print("反向对照(可选): 用一个从未写过的前缀请求, 应看到 [L3-MISS] / prefetch SKIP")
    print("若只有 [L3-MISS] 或走 L2 load_back, 说明 P 在 L1/L2 没被淘汰干净,")
    print("可加大干扰数或调小 --mem-fraction-static / --hicache-ratio。")


if __name__ == "__main__":
    main()
