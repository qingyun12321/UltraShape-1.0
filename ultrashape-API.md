# kokoni-shape API

## 1. 接入概览

`kokoni-shape` 采用异步任务模式：

1. 提交 kokoni-shape 细化任务
2. 获取 `task_id`
3. 轮询任务状态
4. 任务完成后读取结果文件 URL

## 2. 服务地址

当前公共服务地址（下文中的 `base_url`）：

```text
http://36.133.236.108:8091
```

## 3. 鉴权

公共 API 使用 **Bearer Token** 机制进行访问控制。客户端需要在 Header 中传递 `Authorization` 字段。

| Header Field | Value Format | 说明 |
|---|---|---|
| `Authorization` | `Bearer <YOUR_API_KEY>` | 请将 `<YOUR_API_KEY>` 替换为实际分配的密钥 |

## 4. 接口清单

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/services/aigc/3d-generation/reconstruction` | 创建 kokoni-shape 细化任务 |
| `GET` | `/api/v1/tasks/{task_id}` | 查询任务状态与结果 |

## 5. 创建细化任务

### 5.1 请求地址

```text
POST http://36.133.236.108:8091/api/v1/services/aigc/3d-generation/reconstruction
```

### 5.2 请求类型

`multipart/form-data`

### 5.3 请求参数

表单中包含两个部分：

| 参数名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request` | string | 是 | JSON 字符串，外层结构固定为 `model / input / parameters` |
| `files` | file[] | 是 | 输入图片文件；当前仅支持单张图片 |

### 5.4 `request` 字段说明

```json
{
  "model": "kokoni-shape",
  "input": {
    "request_id": "optional-client-id"
  },
  "parameters": {
    "precision": "standard",
    "steps": 50,
    "octree_res": 512,
    "num_latents": 4096,
    "chunk_size": 20000,
    "seed": 42,
    "remove_bg": false,
    "scale": 0.99
  }
}
```

#### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model` | string | 是 | 模型名称，当前使用 `kokoni-shape` |
| `input` | object | 是 | 输入参数 |
| `parameters` | object | 否 | 细化参数，不传时使用默认值 |

#### `input` 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `request_id` | string | 否 | 自动生成 | 客户端请求 ID，建议用于链路追踪 |

#### `parameters` 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `precision` | string | 否 | `standard` | 推理预设 |
| `steps` | integer | 否 | 预设值 | 采样步数 |
| `octree_res` | integer | 否 | 预设值 | 八叉树分辨率 |
| `num_latents` | integer | 否 | 预设值 | latent 数量 |
| `chunk_size` | integer | 否 | 预设值 | 分块大小 |
| `seed` | integer | 否 | `42` | 随机种子 |
| `remove_bg` | boolean | 否 | `false` | 是否先移除背景 |
| `scale` | number | 否 | `0.99` | 输出模型归一化比例 |

上传限制：

- 当前仅支持 `1` 张图片
- 支持格式：`PNG`、`JPG`、`JPEG`、`WEBP`、`BMP`

### 5.5 请求示例

```bash
curl --location 'http://36.133.236.108:8091/api/v1/services/aigc/3d-generation/reconstruction' \
  -H 'Authorization: Bearer <YOUR_API_KEY>' \
  -F 'request={
    "model":"kokoni-shape",
    "input":{
      "request_id":"req-001"
    },
    "parameters":{
      "precision":"standard",
      "seed":42,
      "remove_bg":false,
      "scale":0.99
    }
  }' \
  -F 'files=@/path/to/input.png'
```

### 5.6 成功响应示例

```json
{
  "status_code": 200,
  "request_id": "req-001",
  "code": null,
  "message": "",
  "output": {
    "task_id": "44c6f1f6f2ff42889d29aafec6c64a7a",
    "task_status": "PENDING",
    "submit_time": "2026-03-12 10:20:30.456"
  }
}
```

## 6. 查询任务状态

### 6.1 请求地址

```text
GET http://36.133.236.108:8091/api/v1/tasks/{task_id}
```

### 6.2 路径参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `task_id` | string | 是 | 创建任务接口返回的任务 ID |

### 6.3 任务状态说明

| 状态 | 说明 |
|---|---|
| `PENDING` | 任务已创建，等待平台调度 |
| `SCALING` | 平台正在恢复算力或等待可用节点 |
| `RUNNING` | 任务正在执行 kokoni-shape 细化 |
| `SUCCEEDED` | 任务完成，可读取结果 |
| `FAILED` | 任务失败，请查看 `message` |

### 6.4 查询响应字段

响应根级字段固定如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status_code` | integer | 接口状态码 |
| `request_id` | string | 请求 ID |
| `code` | string \| null | 业务码 |
| `message` | string | 状态说明或错误信息 |
| `output` | object | 任务信息与结果 |

`output` 中固定包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | string | 任务 ID |
| `task_status` | string | 任务状态 |
| `submit_time` | string | 提交时间 |
| `scheduled_time` | string \| null | 调度时间 |
| `start_time` | string \| null | 开始执行时间 |
| `end_time` | string \| null | 结束时间 |

任务成功后，`output` 中还会包含以下关键结果字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 运行会话 ID |
| `glb_url` | string | 细化后 GLB 下载地址 |
| `model_url` | string | 与 `glb_url` 等价的兼容字段 |
| `oss_prefix` | string | 本次任务的 OSS 前缀 |
| `artifacts.input_image.oss_key` | string | 输入图片 OSS 路径 |
| `artifacts.input_image.url` | string | 输入图片签名下载地址 |
| `artifacts.glb.oss_key` | string | 输出 GLB 的 OSS 路径 |
| `artifacts.glb.url` | string | 输出 GLB 的签名下载地址 |

### 6.5 查询示例

```bash
curl --location 'http://36.133.236.108:8091/api/v1/tasks/44c6f1f6f2ff42889d29aafec6c64a7a' \
  -H 'Authorization: Bearer <YOUR_API_KEY>'
```

### 6.6 成功完成响应示例

```json
{
  "status_code": 200,
  "request_id": "req-001",
  "code": null,
  "message": "",
  "output": {
    "task_id": "44c6f1f6f2ff42889d29aafec6c64a7a",
    "task_status": "SUCCEEDED",
    "submit_time": "2026-03-12 10:20:30.456",
    "scheduled_time": "2026-03-12 10:20:35.000",
    "start_time": "2026-03-12 10:20:38.200",
    "end_time": "2026-03-12 10:22:01.123",
    "session_id": "req-001",
    "glb_url": "https://example.com/refined.glb",
    "model_url": "https://example.com/refined.glb",
    "oss_prefix": "docker-input&output/kokoni-shape/req-001",
    "artifacts": {
      "input_image": {
        "oss_key": "docker-input&output/kokoni-shape/req-001/input/input.png",
        "url": "https://example.com/input.png?..."
      },
      "glb": {
        "oss_key": "docker-input&output/kokoni-shape/req-001/output/refined.glb",
        "url": "https://example.com/refined.glb?..."
      }
    }
  }
}
```

## 7. 推荐调用方式

建议客户端按以下方式接入：

1. 调用创建任务接口，获取 `task_id`
2. 每 `2` 秒轮询一次任务状态接口
3. 当 `task_status=SUCCEEDED` 时读取 `output.glb_url`
4. 当 `task_status=FAILED` 时展示 `message`

## 8. 错误处理建议

常见场景如下：

| 场景 | 建议处理方式 |
|---|---|
| `401 Unauthorized` | 检查 Bearer Token 是否缺失或无效 |
| 请求参数不合法 | 根据接口返回信息修正请求后重试 |
| 文件为空或格式不支持 | 检查上传内容 |
| 任务长时间处于 `SCALING` | 平台正在准备可用算力，可继续轮询 |
| 任务返回 `FAILED` | 展示 `message`，并根据业务决定是否重新提交 |
| 查询接口返回 `404` | 检查 `task_id` 是否正确 |

## 9. Runtime 兼容接口

以下接口主要用于内部调试、旧版接入或 runtime 直连，不建议作为新的公共接入方式：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 运行时健康检查 |
| `POST` | `/reconstruct` | task-manager 转发到 runtime 的内部兼容入口 |
| `POST` | `/run_with_files` | 旧版排队接口 |
| `GET` | `/queue_status` | 旧版排队状态接口 |
| `GET` | `/request_status` | 旧版请求状态接口 |
| `POST` | `/generate` | 直出 GLB 下载跳转 |
| `POST` | `/generate_3d` | base64 入参的同步接口 |
| `POST` | `/api/v1/services/aigc/3d-refine/generation` | runtime 自身的异步接口 |
