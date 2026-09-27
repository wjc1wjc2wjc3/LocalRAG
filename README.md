# LocalRAG

**全离线、可审计溯源、带文档级权限的本地 RAG 知识库。**
核心索引 / 检索 / 存储 / 审计全流程 **零第三方依赖**（仅 Python 标准库），一条命令即可运行。

> 许可：**AGPL-3.0**（与 [gitroomhq/postiz-app](https://github.com/gitroomhq/postiz-app/blob/main/LICENSE) 一致）

---

## 一、为什么要再造一个 RAG？

我们对 GitHub 上 **5,844 个 stars > 1000 的 AI 开源项目**（穷举 18 个 AI topic、跨 topic 去重）做了结构统计，结果暴露了一个明显的**供给洼地**：

| 观察项 | 统计结果 | 含义 |
|---|---|---|
| 含「本地 / 自托管 / 离线」的项目 | **仅 83 个（1.4%）**，中位数 star **≈1,400**（大盘中位数 2,403）| 真离线的方案极少，且普遍做得浅 |
| 含「隐私 / 安全 / 审计」的项目 | **仅 62 个（1.1%）** | 可审计几乎是空白 |
| 明确依赖云端大模型 API 的项目 | **139 个，中位数 star 1,371** | 数据出网、API 一变就废，star 也被压制 |
| 名字含 `rag` 的项目 | **仅 19 个** | 且多为 demo，缺生产化能力 |
| star 分布 | **84% 项目 < 5k**，>10k 仅 13.9% | 绝大多数缺 CI / 测试 / 长期维护 |
| 推理 / 部署 / 服务化 | **仅 3.4%** | 工程化能力薄弱 |

**结论**：社区不缺「能跑起来的 RAG demo」，缺的是**敢把公司文档放进去**的 RAG——离线、可审计、有权限、可维护。LocalRAG 就是冲着这个缺口做的。

---

## 二、LocalRAG 有、而现有项目普遍没有的能力

| # | 能力 | 现有项目的普遍情况 | LocalRAG 的做法 |
|---|---|---|---|
| 1 | **全离线 embedding** | 多数方案默认调云端 embedding API，或默认下载远程模型 | 默认 `HashingEmbedder`（纯标准库、零下载、零出网）；语义嵌入可选 `sentence-transformers`，但强制 `local_files_only` 走本地缓存 |
| 2 | **确定性嵌入（可复现）** | 用内置 `hash()` 做特征哈希 → `PYTHONHASHSEED` 随机化导致**重启后向量不可复现**，索引悄悄失效 | 用 `blake2b` 稳定哈希，跨进程 / 跨机器结果一致 |
| 3 | **可审计溯源** | 「隐私/审计」类仅 1.1%；答案通常只有文本，无法回溯来源 | 每个答案强制返回 chunk 级引用：**文档 + 页码 + 字符区间 + 相关度分数** |
| 4 | **哈希链审计日志** | 基本没有（少数项目只打普通 log，可随意删改）| `audit.jsonl` 每条记录串联 `prev_hash`，任何删改都会让 `audit-verify` 失败 |
| 5 | **查询原文不落库** | 日志里直接写明文 query，二次泄露 | 审计只存 `query_sha256` + 长度 + 词数，原文不落盘 |
| 6 | **增量索引** | 很多项目每次全量重建，文档一多就不可用时 | 按文件内容哈希比对，未变更直接 `skipped` |
| 7 | **文档级权限（ACL）** | 几乎都是「全库可见」，多租户场景下不可用 | 每个文档带 `acl` 标签，检索时按标签集合过滤，越权结果为 0 |
| 8 | **被遗忘权** | 删除文档常残留向量 | `forget` 一并删除文档元数据与全部 chunk 向量 |
| 9 | **单文件、零外部服务** | 动辄需要 Postgres / Milvus / Redis 才能跑 | 单文件 SQLite，无服务进程，拷走即迁移 |
| 10 | **零依赖可跑 + 工程可信度** | 84% 项目 star<5k，缺测试与维护 | 核心仅标准库；内置 `unittest` 测试覆盖索引/增量/权限/审计/删除；配置指纹写进审计，保证「同配置可复现」 |

---

## 三、快速开始

```bash
# 核心零依赖，无需安装任何包（可选依赖见 requirements.txt）
py -m localrag.cli index ./docs --acl team
py -m localrag.cli query "报销需要提交什么材料？" --acl team
py -m localrag.cli stats
py -m localrag.cli audit-verify
```

零依赖示例（会自建临时文档并演示完整链路）：

```bash
py examples/quickstart.py
```

跑测试：

```bash
py -m unittest discover -s tests -v
```

---

## 四、Python API

```python
from localrag.config import Config
from localrag.rag import LocalRAG

cfg = Config(db_path="kb.db", audit_path="audit.jsonl", chunk_size=800, chunk_overlap=120)
rag = LocalRAG(cfg)

rag.index_paths(["./docs/hr", "./docs/finance"], acl="team")

ans = rag.ask("年假怎么申请？", acls=["team"])
print(ans.text)                       # 带 [1][2] 引用标记的答案
for c in ans.citations:               # chunk 级溯源
    print(c.location(), c.score)      # hr_policy.txt#p1:120-260  0.42
print(ans.audit_hash)                 # 本次查询的审计记录哈希

print(rag.audit_verify())             # (True, '审计链完整，共 N 条记录')
```

---

## 五、离线是怎么保证的

- 默认 `offline_only=True`：会同时设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
  `HF_HUB_DISABLE_TELEMETRY=1`，把第三方库的**隐式联网与遥测**一起关掉；
- 嵌入、向量计算、存储、生成（抽取式）全程本地，检索期**零出网**；
- 只有显式传 `--allow-network` 才允许联网（例如首次拉取本地语义模型）；
- 无任何埋点 / 遥测代码。

想验证：断网后执行 `py -m localrag.cli query "..."` 依然可用。

---

## 六、可选：接本地模型生成答案

默认 `ExtractiveGenerator` **不需要任何 LLM**（抽取式作答 + 引用标注）。
若想让本地小模型来组织语言，可配合同系列的 **CPUEdgeInference**（本地 CPU 推理服务），
流量只在 `127.0.0.1` 内循环：

```python
from localrag.generator import OpenAICompatGenerator
from localrag.rag import LocalRAG

rag = LocalRAG(generator=OpenAICompatGenerator(
    base_url="http://127.0.0.1:8080/v1", model="auto"))
print(rag.ask("总结一下报销流程").text)
```

模型不可用时会自动回退到抽取式答案，不会抛错。

---

## 七、目录结构

```
LocalRAG/
├── localrag/
│   ├── config.py       # 配置 + 离线开关 + 配置指纹
│   ├── chunking.py     # 文档加载与切分（保留页码/字符区间）
│   ├── embeddings.py   # 确定性 hashing 嵌入 / 可选本地语义嵌入
│   ├── store.py        # 单文件 SQLite（文档 + chunk + 向量 + ACL）
│   ├── audit.py        # 哈希链审计日志（防篡改、可校验）
│   ├── generator.py    # 抽取式生成 / 本地 OpenAI 兼容生成
│   ├── rag.py          # 主流程：索引 → 检索 → 生成 → 审计
│   ├── cli.py          # 命令行
│   └── server.py       # 可选 HTTP API（fastapi）
├── tests/              # unittest 测试
├── examples/           # 零依赖示例
├── LICENSE             # AGPL-3.0
└── README.md
```

---

## 八、路线图

- [ ] 混合检索（向量 + BM25）与重排
- [ ] 多模态：图片 / 表格抽取（仍保持离线）
- [ ] 审计日志导出与可视化（按文档 / 用户维度）
- [ ] 更细粒度的权限（chunk 级 + 字段级）
- [ ] 索引快照与迁移工具
- [ ] 可选 Web UI（本地静态页，无外链资源）

---

## 九、支持本项目

本项目采用 **AGPL-3.0**：你可以自由自托管使用（功能不阉割），
若以云服务形式对外提供，则需同样开源。

如果它帮你把文档安全地留在了自己的机器上，欢迎：

- ⭐ 给仓库点星（最直接的曝光支持）
- 💚 GitHub Sponsors / 一次性赞助（用于长期维护与 issue 响应）
- 🐛 提交 issue / PR（尤其是离线场景与权限模型的边界用例）

---

## License

AGPL-3.0 —— 完整文本见 [LICENSE](./LICENSE)，
与 [postiz-app](https://github.com/gitroomhq/postiz-app/blob/main/LICENSE) 所采用的许可一致。
