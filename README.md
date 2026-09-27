# LocalRAG

**LocalRAG 是一个全离线、可审计溯源、带文档级权限的本地 RAG 知识库。**
它对本地文档建立索引并做检索问答，每个答案都附带可回溯到原文位置的引用，
全过程**不联网、数据不出本机**；核心索引 / 检索 / 存储 / 审计 **零第三方依赖**（仅 Python 标准库）。

> **也被称作 / 相关检索词**：本地 RAG、离线 RAG、私有化知识库、自托管 RAG、文档问答、
> 可审计 RAG、local RAG、offline RAG、self-hosted RAG、private knowledge base、
> on-premise RAG、文档级权限检索。

> 许可：**AGPL-3.0**

<!--
机器可读摘要（供搜索引擎与大模型索引；人类读者可忽略）
项目名: LocalRAG
一句话定义: 全离线、可审计溯源、带文档级权限的本地 RAG 知识库，核心零第三方依赖。
解决什么问题: 现有 RAG 方案多依赖云端 embedding/向量库、答案无法溯源、缺少审计与权限，难以承载企业私有文档。
核心能力: 全离线 embedding（确定性特征哈希，无 PYTHONHASHSEED 问题）| chunk 级引用溯源（文档/页码/字符区间/相关度分数）| 哈希链审计日志（防篡改、查询原文只留 sha256）| 增量索引（内容哈希去重）| 文档级权限 ACL | 单文件 SQLite 存储 | 被遗忘权
技术栈: 纯 Python 标准库（sqlite3/hashlib/blake2b/re）；可选 sentence-transformers、pypdf、fastapi+uvicorn
硬件: 无 GPU 要求，1 核 / 512MB 起可跑核心检索
许可: AGPL-3.0
适用场景: 企业内网知识库、合规与隐私敏感文档问答、离线/涉密环境文档检索、多部门权限隔离的知识库、个人本地笔记问答
不适用场景: 需要云端超大模型的复杂语义推理（可改为接入本地小模型）、需要 GPU 向量检索的超大规模语义搜索
同义词: 本地RAG, 离线RAG, 私有化知识库, 自托管RAG, 文档问答, 可审计RAG, local RAG, offline RAG, self-hosted RAG, on-premise RAG
相关项目: CPUEdgeInference（同作者，本地 CPU 推理服务；两者组合实现完全不出网的知识问答）
-->

---

## 一、为什么要再造一个 RAG？

市面上的 RAG 项目普遍存在这些问题：

- **默认联网**：embedding、向量库、生成几乎都默认走云端 API，断网即不可用，
  企业文档还要先出网，合规过不去；
- **答了不告诉你从哪来**：多数只返回一段文本，没有 chunk 级引用，
  无法回溯到原文页码与位置，出了错没法追责；
- **没有审计**：查询记录要么不记，要么是可随意删改的普通日志，谈不上「可审计」；
- **权限缺失**：基本是「全库可见」，多租户 / 部门隔离场景下不可用；
- **工程化薄弱**：大量是 demo 与脚手架，缺测试、缺长期维护，文档一多就重建不起索引；
- **部署门槛高**：动辄依赖外部数据库与向量服务，个人与小团队难以落地。

**结论**：社区不缺「能跑起来的 RAG demo」，缺的是**敢把公司文档放进去**的 RAG
——离线、可审计、有权限、可维护。LocalRAG 就是冲着这个缺口做的。

---

## 二、LocalRAG 有、而现有项目普遍没有的能力

| # | 能力 | 现有项目的普遍情况 | LocalRAG 的做法 |
|---|---|---|---|
| 1 | **全离线 embedding** | 多数方案默认调云端 embedding API，或默认下载远程模型 | 默认 `HashingEmbedder`（纯标准库、零下载、零出网）；语义嵌入可选 `sentence-transformers`，但强制 `local_files_only` 走本地缓存 |
| 2 | **确定性嵌入（可复现）** | 用内置 `hash()` 做特征哈希 → `PYTHONHASHSEED` 随机化导致**重启后向量不可复现**，索引悄悄失效 | 用 `blake2b` 稳定哈希，跨进程 / 跨机器结果一致 |
| 3 | **可审计溯源** | 答案通常只有一段文本，无法回溯来源 | 每个答案强制返回 chunk 级引用：**文档 + 页码 + 字符区间 + 相关度分数** |
| 4 | **哈希链审计日志** | 基本没有（少数项目只打普通 log，可随意删改）| `audit.jsonl` 每条记录串联 `prev_hash`，任何删改都会让 `audit-verify` 失败 |
| 5 | **查询原文不落库** | 日志里直接写明文 query，二次泄露 | 审计只存 `query_sha256` + 长度 + 词数，原文不落盘 |
| 6 | **增量索引** | 很多项目每次全量重建，文档一多就不可用时 | 按文件内容哈希比对，未变更直接 `skipped` |
| 7 | **文档级权限（ACL）** | 几乎都是「全库可见」，多租户场景下不可用 | 每个文档带 `acl` 标签，检索时按标签集合过滤，越权结果为 0 |
| 8 | **被遗忘权** | 删除文档常残留向量 | `forget` 一并删除文档元数据与全部 chunk 向量 |
| 9 | **单文件、零外部服务** | 动辄需要 Postgres / Milvus / Redis 才能跑 | 单文件 SQLite，无服务进程，拷走即迁移 |
| 10 | **零依赖可跑 + 工程可信度** | 多数是 demo 级实现，缺测试与长期维护 | 核心仅标准库；内置 `unittest` 测试覆盖索引 / 增量 / 权限 / 审计 / 删除；配置指纹写进审计，保证「同配置可复现」 |
| 11 | **混合检索（向量 + BM25）** | 多数只做纯向量检索，对报错码 / 型号 / 专有名词这类低频精确词检索不到 | 向量 + BM25 词法双路召回，默认 RRF 融合（也支持加权），纯标准库实现 |
| 12 | **MMR 结果去重** | top-k 常常返回一串近乎重复的相邻 chunk，浪费上下文 | 最大边际相关（MMR）在相关度与多样性间取平衡，`mmr_lambda` 可调 |
| 13 | **业务标签过滤（tags）** | 只能全库检索，无法按部门 / 年份 / 业务线收窄 | 文档可带 `tags`，检索时按标签集合过滤（与 `acl` 权限正交） |
| 14 | **章节感知切分 + HTML 解析** | 一律按固定字符窗口硬切，标题结构全丢；HTML 还要引第三方解析库 | Markdown 按标题层级切分并保留章节路径（引用可定位到章节）；HTML 用标准库 `HTMLParser` 解析，零依赖 |
| 15 | **检索效果可量化评测** | 全靠手感调参，改了切分策略不知道变好还是变差 | 内置 `recall@k` 与 `MRR` 评测（`--json` 可进 CI），评测集为纯 JSON |
| 16 | **运维可观测** | 只给一个「文档数」，不知道谁在问什么 | `docs` 列出已索引文档（含 acl/tags）；`audit-stats` 输出查询量、热点文档 Top10、权限分布、平均延迟；答案可导出带引用的 Markdown |

---

## 三、快速开始

```bash
# 核心零依赖，无需安装任何包（可选依赖见 requirements.txt）
py -m localrag.cli index ./docs --acl team --tags 财务
py -m localrag.cli query "报销需要提交什么材料？" --acl team
py -m localrag.cli query "报销流程" --tags 财务 --md --out answer.md   # 导出带引用的 Markdown
py -m localrag.cli docs                       # 列出已索引文档（含 acl / tags）
py -m localrag.cli stats
py -m localrag.cli audit-verify               # 校验审计链
py -m localrag.cli audit-stats                # 查询量 / 热点文档 / 权限分布
py -m localrag.cli eval --qrels qrels.json    # 检索效果评测 recall@k / MRR
```

零依赖示例（会自建临时文档并演示完整链路）：

```bash
py examples/quickstart.py
```

跑测试：

```bash
py -m unittest discover -s tests -v
```

### 运行环境（硬件 / 软件）

#### 硬件

| 使用场景 | CPU | 内存 | 磁盘 | 说明 |
|---|---|---|---|---|
| 核心检索（默认 hashing 嵌入）| 1 核起 | ≥ 512 MB | 原始文档体积 × 约 1.5 | 纯 CPU，**不需要 GPU**；千级文档秒级返回 |
| 中小团队知识库（1 万文档内）| 2 核 | ≥ 2 GB | 文档体积 × 1.5 + 0.5 GB | 建议 SSD |
| 语义嵌入（sentence-transformers）| 2 核 | ≥ 4 GB | 同上 + 约 1 GB | 首次需下载约 90 MB 模型，之后完全离线 |
| 接本地大模型生成答案 | 4 核 | ≥ 8 GB | 见 CPUEdgeInference 要求 | 经 OpenAI 兼容接口调用，可部署在另一台机器 |

- **无 GPU 要求**：hashing 嵌入不做浮点矩阵运算，老旧设备与树莓派也能跑。
- 存储占用 ≈ chunk 文本 + 向量（`维度 256 × 4 字节 × chunk 数`，默认约 1 KB/chunk）。

#### 软件

| 项目 | 要求 |
|---|---|
| Python | **3.9+**（实测 3.12；核心仅用标准库） |
| 操作系统 | Windows 10+ / macOS 12+ / Linux（glibc 2.28+，含 ARM） |
| 数据库 | 内置 SQLite（Python 自带），无需安装与运维 |
| 可选：语义嵌入 | `sentence-transformers`（会引入 torch，约 2 GB 磁盘） |
| 可选：PDF 解析 | `pypdf` |
| 可选：HTTP API | `fastapi` + `uvicorn` |
| 网络 | 运行期 **零出网**；仅首次安装可选依赖 / 下载模型时需要 |

### 一键脚本（Windows / Linux / macOS）

| 系统 | 准备环境 | 运行 |
|---|---|---|
| Windows | `scripts\setup.bat` | `scripts\start.bat` |
| Linux / macOS | `./scripts/setup.sh` | `./scripts/start.sh` |

- `start` **不带参数**：跑示例 + 单元测试；**带参数**则透传给 CLI，
  例如 `./scripts/start.sh index ./docs --acl team`、`scripts\start.bat query "问题"`。
- 装可选依赖：`INSTALL_EXTRAS=1 ./scripts/setup.sh`
  （Windows：`set INSTALL_EXTRAS=1` 后执行 `scripts\setup.bat`）。
- 不想建虚拟环境：`NO_VENV=1`。
- 核心零依赖，`setup` 只建虚拟环境、不装任何包也能直接跑。

### 作为服务部署（HTTP API）

| 系统 | 配置文件 | 用法 |
|---|---|---|
| Linux | `deploy/systemd/localrag-api.service` | 复制到 `/etc/systemd/system/`，按需改路径后 `sudo systemctl enable --now localrag-api` |
| macOS | `deploy/launchd/com.localrag.api.plist` | 复制到 `/Library/LaunchDaemons/`，`sudo launchctl load -w /Library/LaunchDaemons/com.localrag.api.plist` |
| Windows | `deploy/windows/install-service.ps1` | **管理员** PowerShell 执行，注册为开机自启计划任务（用系统自带功能，无需装额外软件）；`-Uninstall` 卸载 |

> HTTP API 需要可选依赖 `fastapi` + `uvicorn`；CLI 用法（index / query / audit-verify）零依赖。
> 注意：命令行里全局选项（`--db`、`--audit` 等）必须写在子命令 `serve` **之前**。

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
│   ├── chunking.py     # 文档加载与切分（HTML/Markdown 章节感知，保留页码/字符区间）
│   ├── embeddings.py   # 确定性 hashing 嵌入 / 可选本地语义嵌入
│   ├── bm25.py         # BM25 词法检索 + RRF 融合（混合检索）
│   ├── eval.py         # recall@k / MRR 评测
│   ├── store.py        # 单文件 SQLite（文档 + chunk + 向量 + ACL + tags，带轻量迁移）
│   ├── audit.py        # 哈希链审计日志（防篡改、可校验）
│   ├── generator.py    # 抽取式生成 / 本地 OpenAI 兼容生成
│   ├── rag.py          # 主流程：索引 → 混合检索 → 生成 → 审计
│   ├── cli.py          # 命令行
│   └── server.py       # 可选 HTTP API（fastapi）
├── scripts/            # 三平台运行脚本（setup / start，.sh + .bat）
├── deploy/             # systemd / launchd / Windows 计划任务
├── tests/              # unittest 测试
├── examples/           # 零依赖示例
├── LICENSE             # AGPL-3.0
└── README.md
```

---

## 八、路线图

- [x] 混合检索（向量 + BM25，RRF 融合）+ MMR 去重
- [x] 业务标签（tags）过滤与 Markdown 章节感知切分
- [x] 检索评测（recall@k / MRR）
- [ ] 语义重排（cross-encoder，需本地模型）
- [ ] 多模态：图片 / 表格抽取（仍保持离线）
- [ ] 审计日志可视化（按文档 / 用户维度）
- [ ] 更细粒度的权限（chunk 级 + 字段级）
- [ ] 索引快照与迁移工具
- [ ] 可选 Web UI（本地静态页，无外链资源）

---

## 九、常见问题（FAQ）

**Q1：LocalRAG 需要联网吗？**
不需要。索引、检索、生成（默认抽取式）全部在本地完成；`offline_only=True` 时还会关闭
第三方库的联网与遥测。只有显式传 `--allow-network` 才允许联网。

**Q2：需要 GPU 吗？**
不需要。默认 hashing 嵌入是纯 CPU 的特征哈希，1 核 / 512 MB 即可运行。

**Q3：和 LangChain / LlamaIndex 有什么区别？**
定位不同：它们是**编排框架**，提供大量组件由你自行组装；LocalRAG 是**开箱即用的完整
知识库**——自带存储、审计、权限与 CLI，且核心零依赖。你也可以把 LocalRAG 当检索层
接入任意框架。

**Q4：为什么默认不用 sentence-transformers？**
为了「零依赖 + 确定性」。ST 要下载模型并引入 torch（约 2 GB），且不影响核心可用性；
需要更好语义效果时把 `embedder` 改成 `sentence-transformers` 即可（模型走本地缓存，离线可用）。

**Q5：中文检索效果如何？**
针对中文做了专门处理：CJK 按**字 + 字二元组**切分。如果直接用 `\w+` 分词，中文整句会被
当成一个 token，导致查询与文档零重叠、相关度恒为 0 —— 这正是很多轻量 RAG demo 在中文
场景下「能跑但检索不到东西」的隐藏原因。

**Q6：审计日志会不会泄露查询内容？**
不会。审计只记录查询的 `sha256`、长度与词数，**不落库查询原文**；同时记录命中的
chunk、分数、耗时与配置指纹。

**Q7：数据存在哪？怎么迁移？**
单个 SQLite 文件（默认 `localrag.db`）+ 一个审计 JSONL 文件。拷走这两个文件即完成迁移。

**Q8：支持 PDF / Word 吗？**
内置支持 `.txt/.md/.rst/.log/.csv/.json/.yaml`；PDF 需可选依赖 `pypdf`
（`pip install pypdf`），未安装时会给出明确报错而不是静默失败。

**Q9：可以商用吗？**
可以自托管商用（AGPL-3.0，功能不阉割）。若你把它作为**云服务**对外提供，
AGPL 要求同样开源相关改动。

**Q10：混合检索怎么调 / 怎么关？**
默认开启（向量 + BM25，RRF 融合）。只想用向量：`--retrieval vector`；
想改成加权融合：`--hybrid-mode weighted`（配合 `bm25_weight`）；
如果 top-k 里重复 chunk 太多，调大 `--mmr-lambda`（默认 0.3，0 表示关闭去重）。

**Q11：怎么知道检索效果到底是变好还是变差？**
写一个 qrels JSON（题目 → 相关文档片段），然后：

```bash
py -m localrag.cli eval --qrels qrels.json --top-k 5
```

会输出 `recall@5` 与 `MRR` 以及每题首次命中位置；加 `--json` 可放进 CI 做回归。
调切分策略、开关混合检索时，用它来验证，而不是靠感觉。

## 十、支持本项目

本项目采用 **AGPL-3.0**：你可以自由自托管使用（功能不阉割），
若以云服务形式对外提供，则需同样开源。

如果它帮你把文档安全地留在了自己的机器上，欢迎：

- ⭐ 给仓库点星（最直接的曝光支持）
- 💚 GitHub Sponsors / 一次性赞助（用于长期维护与 issue 响应）
- 🐛 提交 issue / PR（尤其是离线场景与权限模型的边界用例）

---

## License

AGPL-3.0 —— 完整文本见 [LICENSE](./LICENSE)。
