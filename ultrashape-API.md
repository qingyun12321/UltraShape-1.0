# UltraShape API 文档

适用默认项目：`ultrashape`

## 1. 服务地址

### 1.1 Task-Manager

- 默认地址：`http://36.133.236.108:8090`
- 恢复任务：`POST /api/task/recover`

请求体：

```json
{
  "project": "ultrashape"
}
```

关键响应字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `service_url` | string | 是 | Runtime API 基地址 |
| `recovered` | boolean | 否 | 本次请求是否触发任务恢复 |

失败语义：

- 非 `2xx`：恢复失败
- `service_url` 为空：不可继续后续调用

### 1.2 Runtime API

运行时服务基地址使用 `recover` 返回的 `service_url`。

默认启动端口：`10083`。

对客户开放的接口：

- `GET /health`
- `POST /run_with_files`
- `GET /queue_status`
- `GET /request_status`

说明：

- 生成完成后的模型文件不通过 Runtime API 直接返回二进制内容。
- 最终结果通过 `request_status` 返回的 `result.artifacts.glb.url` 提供下载与预览。
- 服务空闲后会自动释放算力，客户侧无需额外调用暂停接口。

## 2. 调用顺序

1. `POST {task_manager}/api/task/recover`
2. `GET {service_url}/health`，轮询至可用
3. `POST {service_url}/run_with_files`
4. 轮询任务状态
   - `GET {service_url}/queue_status?request_id=...`
   - `GET {service_url}/request_status?request_id=...`
5. 读取 `result.artifacts.glb.url`，用于下载或在线预览

## 3. 接口定义

### 3.1 `GET /health`

请求参数：无

响应示例：

```json
{
  "status": "ok"
}
```

### 3.2 `POST /run_with_files`

请求类型：`multipart/form-data`

用途：上传单张图片并创建 3D 生成任务。

#### 3.2.1 表单参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | file | 是 | 无 | 输入图片，仅支持单张 |
| `request_id` | string | 否 | 自动生成 | 客户端请求 ID，建议使用 UUID |
| `precision` | string | 否 | `standard` | 推理预设 |
| `steps` | integer | 否 | 预设值 | 推理步数 |
| `octree_res` | integer | 否 | 预设值 | 八叉树分辨率 |
| `num_latents` | integer | 否 | 预设值 | Latent 数量 |
| `chunk_size` | integer | 否 | 预设值 | 分块大小 |
| `seed` | integer | 否 | `42` | 随机种子 |
| `remove_bg` | boolean | 否 | `false` | 是否移除背景 |
| `scale` | float | 否 | `0.99` | 模型归一化比例 |

上传限制：

- 单次仅支持 `1` 张图片
- 支持格式：`PNG`、`JPG`、`JPEG`、`WEBP`、`BMP`

成功响应：

```json
{
  "status": "queued",
  "request_id": "0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377",
  "position": 1
}
```

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 固定为 `queued` |
| `request_id` | string | 请求 ID |
| `position` | integer | 当前排队位置，`1` 表示队首 |

错误码：

- `400`：上传文件为空、类型不合法，或上传数量不符合要求
- `409`：`request_id` 与队列中的待处理/处理中任务冲突
- `422`：表单参数校验失败
- `429`：队列已满

请求示例：

```bash
curl -X POST "http://127.0.0.1:10083/run_with_files" \
  -F "image=@/path/to/input.png" \
  -F "request_id=req-001" \
  -F "precision=standard"
```

### 3.3 `GET /queue_status`

Query 参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 否 | 指定后返回该请求的排队信息 |

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `processing` | boolean | 当前是否存在执行中的任务 |
| `pending` | integer | 当前等待队列中的任务数 |
| `current_request_id` | string | 当前执行中的请求 ID，无则为空字符串 |
| `status` | string | 队列状态或指定请求状态 |
| `position` | integer | 指定 `request_id` 时返回 |

`status` 取值说明：

- 未传 `request_id`：`processing` / `pending` / `idle`
- 传入 `request_id`：`pending` / `processing` / `completed` / `failed` / `unknown`

`position` 规则：

- `0`：请求正在处理
- `>=1`：请求在等待队列中，`1` 表示队首
- `-1`：请求不存在，或已不在等待队列中

响应示例：

```json
{
  "processing": true,
  "pending": 2,
  "current_request_id": "e5ab13d4-2f13-4a26-a6a4-bf2bc0f151f4",
  "status": "pending",
  "position": 1
}
```

### 3.4 `GET /request_status`

Query 参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 是 | 请求 ID |

错误码：

- `400`：`request_id` 为空
- `404`：未找到对应请求

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | string | 请求 ID |
| `status` | string | `pending` / `processing` / `completed` / `failed` |
| `error` | string | 失败原因，成功时为空字符串 |
| `created_at` | number | 创建时间，Unix 时间戳（秒） |
| `started_at` | number \| null | 开始处理时间，Unix 时间戳（秒） |
| `finished_at` | number \| null | 处理结束时间，Unix 时间戳（秒） |
| `result` | object | 结果对象，任务完成后返回 |

`result.artifacts` 关键字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `input_image.oss_key` | string | 输入图片 OSS 路径 |
| `input_image.url` | string | 输入图片签名访问地址 |
| `glb.oss_key` | string | 输出 GLB 的 OSS 路径 |
| `glb.url` | string | 输出 GLB 的签名下载地址 |

成功响应示例：

```json
{
  "request_id": "0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377",
  "status": "completed",
  "error": "",
  "created_at": 1773086400.123,
  "started_at": 1773086402.581,
  "finished_at": 1773086468.004,
  "result": {
    "request_id": "0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377",
    "oss_prefix": "docker-input&output/ultrashape/0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377",
    "artifacts": {
      "input_image": {
        "oss_key": "docker-input&output/ultrashape/0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377/input/input.png",
        "url": "https://example.com/input.png?..."
      },
      "glb": {
        "oss_key": "docker-input&output/ultrashape/0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377/output/refined.glb",
        "url": "https://example.com/refined.glb?..."
      }
    }
  }
}
```

失败响应示例：

```json
{
  "request_id": "0bbf1d97-2b20-4a1f-8d2c-5a9efbd1f377",
  "status": "failed",
  "error": "Queue is full, try again later",
  "created_at": 1773086400.123,
  "started_at": null,
  "finished_at": 1773086400.456,
  "result": {}
}
```

## 4. 客户端轮询建议

- 提交任务后立即保存 `request_id`
- 以 `2` 秒间隔轮询 `queue_status` 与 `request_status`
- `request_status.status=completed` 后，直接读取 `result.artifacts.glb.url`
- `request_status.status=failed` 时，以 `error` 字段作为失败原因展示

## 5. 前端对接说明

`3d_preview.html` 的默认行为：

- 页面加载后不主动恢复任务
- 用户点击“开始生成”时调用 `recover`
- 后端健康检查通过后提交 `run_with_files`
- 轮询状态直至获取 `glb.url`
- 使用 `glb.url` 加载模型预览，并将同一地址用于“下载当前 GLB”
