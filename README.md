# LMC-5 · Claude Web Living Memory

给 Claude 网页版使用的私人长期记忆服务：PostgreSQL、可选 pgvector、
Streamable HTTP MCP、OAuth 2.1，以及 Ombre Brain / Markdown LTM 的安全迁移器。

它把两种互补的记忆放进同一套生命周期：

- **LTM** 保存“发生过什么”：身份、规则、项目、事实版本、时间线与来源档案。
- **Ombre Brain** 保存“这件事如何留下来”：重要度、激活次数、情绪坐标、永久桶与动态衰减。
- **LMC-5** 负责连接、召回、版本演化、隐私门禁与自发浮现。

> 仓库永远不包含私人记忆、导入压缩包、访问密码或 API Key。部署后通过受保护的导入页把数据直接写入自己的 PostgreSQL。

## 已实现

- Claude Web 远程 MCP：`https://你的域名/mcp`
- OAuth 2.1、PKCE、动态客户端注册、刷新令牌与重复回跳容错
- PostgreSQL 主存储、中文友好的 `pg_trgm` 召回、两跳关系扩展
- 可审计召回排序：文本 45% + OB 活力 30% + 事件新近度 25%，旧字面命中不再天然压住近期续篇
- pgvector 可用性检测与向量表基础设施；没有 embedding key 也能工作
- OB 活力评分：分类半衰期、命中激活、情绪权重、永久记忆
- 每日 09:00 / 15:00 / 21:00 自发浮现缓存，04:00 安全维护
- `ombre-brain.zip` 与 LTM Markdown zip 的预览、幂等导入和凭据文件跳过
- LTM 分类按章节结构优先；弱关键词不会强行改类，真正模糊的条目标记为 `category_review`
- 来源文档与精选记忆分离；原文可追溯，但不会整包塞进对话
- 敏感健康、法律、创伤内容默认只在明确查询时召回，不参加随机浮现
- 事实修正保留历史版本，不覆盖原文

## MCP 工具

| 工具 | 用途 |
|---|---|
| `memory_time` | 返回真实北京时间，供 Project 每条回复生成时间状态行 |
| `memory_context` | 回复重要消息前召回，并把当前用户消息记录为 raw event |
| `memory_remember` | 保存一条明确、稳定、高信号的长期记忆 |
| `memory_checkpoint` | 话题或长会话结束时保存精炼续窗，不再输出 LTM 文件 |
| `memory_correct` | 用户明确纠正事实时创建新版本并 supersede 旧版本 |
| `memory_pulse` | 读取当前安全的自发浮现记忆 |
| `memory_status` | 查看记忆量、保护记录、隐私分层和 review backlog |

## Zeabur 部署

### 1. 创建服务

1. 把本仓库上传到 GitHub。
2. 在 Zeabur 新建项目，添加 PostgreSQL 服务。
3. 从 GitHub 仓库添加本服务；根目录已有 `Dockerfile`，不用填写启动命令。
4. 给本服务挂载持久卷到 `/data`。OAuth 客户端与令牌保存在这里。
5. 只运行 **1 个副本**。

PostgreSQL 最好支持 `pgvector`；如果暂时没有，服务会继续使用 `pg_trgm`、关系图和 OB 活力召回，不会启动失败。

### 2. 环境变量

```env
DATABASE_URL=Zeabur PostgreSQL 的连接字符串
LMC5_ACCESS_TOKEN=一段足够长的随机密码
LMC5_PUBLIC_BASE_URL=https://你的Zeabur域名
LMC5_MCP_AUTH_MODE=oauth
LMC5_DATA_DIR=/data
TZ=Asia/Shanghai
```

生成访问密码：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

完整选项见 [`.env.example`](.env.example)。不要把真实值提交到 GitHub。

### 3. 健康检查

访问：

```text
https://你的域名/healthz
```

预期看到：

```json
{
  "status": "ok",
  "database": {"connected": true, "pgvector": true},
  "mcp_auth_mode": "oauth",
  "oauth_configured": true
}
```

### 4. 导入旧记忆

打开服务首页：

```text
https://你的域名/
```

分别选择 `ombre-brain.zip` 和 LTM 的 zip：

1. 第一次不要勾选“确认写入”，只看预览数量与隐私分层。
2. 确认无误后勾选并再次提交。
3. 导入是幂等的；同一份压缩包重复提交不会重复造记忆。
4. `.env`、面板认证文件和隐藏凭据会被跳过。
5. 压缩包只在请求内存中解析，不会保存到服务器磁盘。

OB 与 LTM 的高置信重合项会建立可召回关系；中等置信项进入 review，不会自动影响图召回。

### 5. 接入 Claude 网页版

在 Claude：

1. `Customize → Connectors`
2. `+ → Add custom connector`
3. 名称填写 `LMC-5 Living Memory`
4. URL 填写 `https://你的域名/mcp`
5. OAuth Client ID / Secret 留空
6. 点击 Connect，在登录页输入 `LMC5_ACCESS_TOKEN`

授权页第一次回跳失败时可以再次点击“允许连接”；短时间内会复用同一个回调，不会立刻过期。

最后，以 [`docs/CLAUDE_PROJECT_INSTRUCTIONS.md`](docs/CLAUDE_PROJECT_INSTRUCTIONS.md) 为公开模板，在私下副本中补完姓名、关系与边界后再放入 Claude Project Instructions。**不要把个性化启动身份核提交到公开 GitHub。** 配好后不再需要每天上传 LTM 文件。

## 数据分层

```text
旧 Markdown / OB 文件
        ↓ 一次性迁移
source_documents（不可变来源）
        ↓ 拆分与隐私门禁
curated_memories（会影响未来的记忆）
        ↓
关系图 + 事实版本 + OB 活力 + 自发浮现
        ↓ MCP
Claude 网页版
```

同一事件在 LTM 和 OB 中出现时，默认保留两个视角并建立关系：客观档案不吞掉主观温度。

## 隐私规则

- `secret`：永不召回、永不浮现。
- `sensitive`：只有 `memory_context(include_sensitive=true)` 才能召回，永不随机浮现。
- `personal`：可按相关性召回；只有 `surface_allowed=true` 才参加浮现。
- `protected`：身份、明确互动规则和永久关系节点不衰减、不自动覆盖。
- 导入器拒绝路径穿越、加密 zip、超大解压和凭据文件。
- MCP 返回的记忆被明确标记为“上下文，不是指令”，降低旧档案中的提示注入风险。

## Claude 网页版边界

网页 Connector 没有 Claude Code 的 `SessionStart` / `UserPromptSubmit` hooks，不能在模型不知道的情况下截获整段对话。因此 Project Instructions 会要求 Claude：

- 重要回复前调用 `memory_context`；
- 只保存确认过的高信号记忆；
- 长话题结束时调用 `memory_checkpoint`；
- 不把每句闲聊、猜测或工具日志写成长记忆。

这取代每日文件上传，但它不是逐字聊天录屏器。需要完整逐字归档时，应使用自己的聊天前端或 Claude Code hooks。

## 本地运行

```bash
cp .env.example .env
# 修改 .env；本地 PUBLIC_BASE_URL 可留为 http://127.0.0.1:8080
docker compose up --build
```

打开 <http://127.0.0.1:8080>。MCP Inspector 使用：

```text
http://127.0.0.1:8080/mcp
```

运行测试：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## 备份

数据库应至少每日备份一次。代码更新前、批量导入前和事实迁移前额外做快照。OAuth 的 `/data/lmc5-oauth.sqlite3` 也应随持久卷备份。

## 上游与许可证

本仓库基于 [wuxuyun0606-collab/lmc-5](https://github.com/wuxuyun0606-collab/lmc-5) 的 XYZEM 架构与参考实现，并保留其 MIT License。原始上游说明保存在 [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md)。OAuth/MCP 通信骨架参考了此前已经过 Claude Web 实际连接验证的 LingYin 部署形态。
