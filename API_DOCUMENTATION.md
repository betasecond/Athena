# Preparade API 文档

本文档详细介绍了智能客服系统(Copilot)的API接口规范，包括数据结构、请求参数和响应格式。后端开发人员可依照此文档实现对应的API端点。

## 目录

1. [通用规范](#通用规范)
2. [智能问答服务](#智能问答服务)
3. [客服辅助服务](#客服辅助服务)
4. [知识库审核队列服务](#知识库审核队列服务)

## 通用规范

### API响应格式

所有API响应均遵循以下统一格式：

```typescript
{
  "code": 200,          // 状态码，200表示成功
  "message": "成功",    // 响应消息
  "data": {},           // 响应数据，具体格式因API而异
  "requestId": "xxx",   // 请求标识符，用于追踪和调试
  "processTime": 120    // 服务器处理时间(ms)
}
```

### 分页响应格式

需要分页的API响应使用以下格式：

```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "items": [],        // 当前页数据项
    "total": 100,       // 总记录数
    "page": 1,          // 当前页码
    "pageSize": 10,     // 每页记录数
    "totalPages": 10,   // 总页数
    "hasMore": true     // 是否有下一页
  }
}
```

### 错误响应格式

请求失败时的响应格式：

```typescript
{
  "code": 400,          // 错误状态码
  "message": "参数错误", // 错误消息
  "data": null,         // 通常为null或包含错误详情
  "requestId": "xxx"    // 请求标识符
}
```

### 常见状态码

| 状态码 | 描述 |
|--------|------|
| 200    | 成功 |
| 400    | 请求参数错误 |
| 401    | 未授权/鉴权失败 |
| 403    | 权限不足 |
| 404    | 资源不存在 |
| 500    | 服务器内部错误 |

## 智能问答服务

智能问答服务提供对用户查询的处理和智能回复功能。

### 提交用户问题

**接口**: `POST /api/query`

**描述**: 提交用户问题并获取智能回答

**请求参数**:
```typescript
{
  "query": string,           // 用户原始问题内容 (必填)
  "sessionId": string,       // 会话ID (可选)
  "userId": string,          // 用户ID (可选)
  "source": string,          // 查询来源: web, mobile, kiosk 等 (可选)
  "context": object          // 附加上下文信息 (可选)
}
```

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "originalQuery": string,             // 原始查询问题
    "keywordAnalysis": [                 // 识别的关键词分析
      {
        "keyword": string,               // 识别的关键词
        "type": string,                  // 关键词类型：产品、服务等
        "confidence": number             // 置信度得分 (0-1)
      }
    ],
    "knowledgeMatches": [                // 匹配到的知识条目
      {
        "id": string,                    // 知识条目ID
        "standardQuestion": string,      // 标准问题
        "relevanceScore": number,        // 匹配相似度得分 (0-1)
        "lastUpdated": string,           // 知识条目最后更新时间
        "tags": string[]                 // 知识条目标签 (可选)
      }
    ],
    "suggestedAnswer": string,           // 建议的回复内容
    "confidence": number,                // 建议回复的置信度 (0-1)
    "needsHumanReview": boolean,         // 是否触发专业人工介入
    "references": string[],              // 回答的知识来源参考 (可选)
    "relatedQuestions": string[]         // 相关推荐问题 (可选)
  }
}
```

### 提交用户反馈

**接口**: `POST /api/feedback`

**描述**: 提交用户对回答的反馈

**请求参数**:
```typescript
{
  "queryId": string,         // 问题ID (必填)
  "responseId": string,      // 回答ID (必填)
  "feedbackType": "helpful" | "not-helpful" | "partially-helpful",  // 反馈类型 (必填)
  "comment": string,         // 详细反馈内容 (可选)
  "userId": string           // 用户ID (可选)
}
```

**响应**:
```typescript
{
  "code": 200,
  "message": "反馈提交成功",
  "data": {
    "success": true
  }
}
```

### 获取热门问题

**接口**: `GET /api/hot-questions`

**描述**: 获取系统热门问题列表

**查询参数**:
- `page`: 页码，默认为1
- `pageSize`: 每页记录数，默认为10
- `sortBy`: 排序字段 (可选)
- `sortOrder`: 排序方向: asc | desc (可选)

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "items": [
      {
        "id": string,        // 问题ID
        "question": string,  // 问题内容
        "count": number      // 提问次数
      }
    ],
    "total": number,         // 总记录数
    "page": number,          // 当前页码
    "pageSize": number,      // 每页记录数
    "totalPages": number,    // 总页数
    "hasMore": boolean       // 是否有下一页
  }
}
```

### 获取用户历史查询

**接口**: `GET /api/user-history`

**描述**: 获取指定用户的历史查询记录

**查询参数**:
- `userId`: 用户ID (必填)
- `page`: 页码，默认为1
- `pageSize`: 每页记录数，默认为20
- `sortBy`: 排序字段 (可选)
- `sortOrder`: 排序方向: asc | desc (可选)

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "items": [
      {
        "id": string,        // 查询ID
        "query": string,     // 查询内容
        "timestamp": string  // 查询时间
      }
    ],
    "total": number,         // 总记录数
    "page": number,          // 当前页码
    "pageSize": number,      // 每页记录数 
    "totalPages": number,    // 总页数
    "hasMore": boolean       // 是否有下一页
  }
}
```

## 客服辅助服务

客服辅助服务为客服人员提供实时辅助和建议。

### 获取实时辅助建议

**接口**: `POST /api/agent-assist/suggestions`

**描述**: 根据对话上下文和当前输入获取实时辅助建议

**请求参数**:
```typescript
{
  "context": {                           // 当前对话上下文 (必填)
    "sessionId": string,                 // 会话ID
    "customerId": string,                // 客户ID (可选)
    "agentId": string,                   // 客服ID
    "history": [                         // 对话历史
      {
        "id": string,                    // 消息ID
        "sender": "customer" | "agent" | "system",  // 发送者类型
        "content": string,               // 消息内容
        "timestamp": string,             // 发送时间
        "type": "text" | "image" | "file" | "system-notice",  // 消息类型
        "status": "sending" | "sent" | "read" | "failed",     // 消息状态 (可选)
        "sentiment": {                   // 情感分析结果 (可选)
          "type": "positive" | "negative" | "neutral",  // 情感类型
          "score": number                // 情感强度 (0-1)
        }
      }
    ],
    "customerInfo": {                    // 客户信息 (可选)
      "name": string,                    // 客户名称
      "vipLevel": number,                // VIP等级
      "orderCount": number,              // 历史订单数
      "tags": string[]                   // 客户标签
    },
    "orderInfo": {                       // 相关订单信息 (可选)
      "orderId": string,                 // 订单ID
      "status": string,                  // 订单状态
      "amount": number,                  // 订单金额
      "orderDate": string                // 下单时间
    },
    "topics": string[]                   // 对话主题分类 (可选)
  },
  "currentDraft": string,                // 当前正在编辑的回复草稿
  "cursorPosition": number,              // 光标位置 (可选)
  "assistType": "auto" | "completion" | "sentiment" | "compliance"  // 请求辅助的类型 (可选)
}
```

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "suggestions": [                     // 建议列表
      {
        "id": string,                    // 建议ID
        "type": "completion" | "sentiment" | "compliance" | "information" | "greeting" | "closing",  // 建议类型
        "content": string,               // 建议内容
        "priority": number,              // 建议的优先级 1-5，1为最高
        "confidence": number,            // 建议的置信度 (0-1)
        "reason": string,                // 触发建议的原因 (可选)
        "sourceId": string               // 建议来源的知识库ID或规则ID (可选)
      }
    ],
    "sentimentAnalysis": {               // 情感分析 (可选)
      "type": "positive" | "negative" | "neutral",  // 检测到的情感类型
      "score": number,                   // 情感强度 (0-1)
      "keywords": string[]               // 检测到的关键情感词 (可选)
    },
    "complianceAnalysis": {              // 合规性分析 (可选)
      "hasIssues": boolean,              // 是否存在合规问题
      "issues": [                        // 问题列表 (可选)
        {
          "type": string,                // 问题类型
          "description": string,         // 问题描述
          "severity": "low" | "medium" | "high",  // 严重程度
          "suggestion": string           // 建议的修正方式 (可选)
        }
      ]
    },
    "informationCompleteness": {         // 信息完整性检查 (可选)
      "missingInfo": string[],           // 缺失的关键信息
      "suggestedQuestions": string[]     // 收集信息的建议问题
    }
  }
}
```

### 搜索知识库

**接口**: `GET /api/knowledge/search`

**描述**: 搜索知识库内容

**查询参数**:
- `query`: 搜索关键词 (必填)
- `limit`: 最大返回结果数 (可选)
- `tags`: 搜索范围标签，可多选 (可选)

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": [
    {
      "id": string,                  // 知识ID
      "title": string,               // 知识标题
      "content": string,             // 知识内容
      "lastUpdated": string,         // 最后更新时间
      "tags": string[],              // 知识标签 (可选)
      "relevanceScore": number       // 相关性得分
    }
  ]
}
```

### 记录建议使用情况

**接口**: `POST /api/agent-assist/track-usage`

**描述**: 记录客服选择应用的建议

**请求参数**:
```typescript
{
  "suggestionId": string,    // 被应用的建议ID (必填)
  "sessionId": string        // 会话ID (必填)
}
```

**响应**:
```typescript
{
  "code": 200,
  "message": "记录成功",
  "data": {
    "success": true
  }
}
```

### 获取回复模板

**接口**: `GET /api/agent-assist/templates`

**描述**: 获取常用回复模板

**查询参数**:
- `agentId`: 客服ID (必填)
- `category`: 模板类别 (可选)

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": [
    {
      "id": string,          // 模板ID
      "title": string,       // 模板标题
      "content": string,     // 模板内容
      "category": string     // 模板类别
    }
  ]
}
```

## 知识库审核队列服务

知识库审核队列服务管理需要审核的知识条目。

### 获取审核队列列表

**接口**: `GET /api/review-queue/items`

**描述**: 获取审核队列列表

**查询参数**:
- `page`: 页码，默认为1
- `pageSize`: 每页记录数，默认为10
- `sortBy`: 排序字段 (可选)
- `sortOrder`: 排序方向: asc | desc (可选)
- `keyword`: 搜索关键词 (可选)
- `status`: 审核状态: pending | approved | rejected | needsInfo (可选)
- `source`: 来源类型 (可选)
- `tags`: 标签，可多选 (可选)
- `startDate`: 起始日期 (可选)
- `endDate`: 截止日期 (可选)

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "items": [
      {
        "id": string,                    // 审核项ID
        "source": string,                // 来源：用户提问-系统未答出、用户反馈-答案差评等
        "originalQuery": string,         // 原始用户问题
        "currentAnswer": string,         // 系统当前给出的答案 (可选)
        "suggestedAnswer": string,       // 推荐/修正后的答案 (可选)
        "timestamp": string,             // 问题发生或提交审核的时间
        "status": "pending" | "approved" | "rejected" | "needsInfo",  // 状态
        "standardQuestion": string,      // 标准问题 (可选)
        "metadata": {                    // 元数据 (可选)
          "tags": string[],              // 标签
          "keywords": string[],          // 关键词
          "expirationDate": string       // 过期日期
        }
      }
    ],
    "total": number,                     // 总记录数
    "page": number,                      // 当前页码
    "pageSize": number,                  // 每页记录数
    "totalPages": number,                // 总页数
    "hasMore": boolean                   // 是否有下一页
  }
}
```

### 获取审核项详情

**接口**: `GET /api/review-queue/items/{itemId}`

**描述**: 获取单个审核项详细信息

**路径参数**:
- `itemId`: 审核项ID

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "id": string,                  // 审核项ID
    "source": string,              // 来源
    "originalQuery": string,       // 原始用户问题
    "currentAnswer": string,       // 系统当前给出的答案 (可选)
    "suggestedAnswer": string,     // 推荐/修正后的答案 (可选)
    "timestamp": string,           // 问题发生或提交审核的时间
    "status": "pending" | "approved" | "rejected" | "needsInfo",  // 状态
    "standardQuestion": string,    // 标准问题 (可选)
    "metadata": {                  // 元数据 (可选)
      "tags": string[],            // 标签
      "keywords": string[],        // 关键词
      "expirationDate": string     // 过期日期
    },
    "relatedKnowledgeItems": [     // 相关知识条目 (可选)
      {
        "id": string,              // 知识条目ID
        "question": string,        // 标准问题
        "similarityScore": number  // 相似度得分
      }
    ],
    "reviewHistory": [             // 审核历史记录 (可选)
      {
        "reviewerId": string,      // 审核人ID
        "reviewerName": string,    // 审核人名称
        "timestamp": string,       // 审核时间
        "action": string,          // 审核动作
        "comment": string          // 审核备注 (可选)
      }
    ]
  }
}
```

### 提交审核决定

**接口**: `POST /api/review-queue/decision`

**描述**: 提交审核决定

**请求参数**:
```typescript
{
  "itemId": string,                      // 审核项ID (必填)
  "decision": "approve" | "reject" | "needsInfo",  // 审核决定 (必填)
  "standardQuestion": string,            // 标准问题 (批准时必填)
  "suggestedAnswer": string,             // 标准答案 (批准时必填)
  "metadata": {                          // 元数据 (可选)
    "tags": string[],                    // 标签
    "keywords": string[],                // 关键词
    "expirationDate": string             // 过期日期
  },
  "comment": string                      // 审核备注 (可选)
}
```

**响应**:
```typescript
{
  "code": 200,
  "message": "审核完成",
  "data": {
    "success": true,
    "knowledgeItemId": string            // 如果批准，返回创建的知识条目ID
  }
}
```

### 批量操作审核项

**接口**: `POST /api/review-queue/batch-operation`

**描述**: 批量操作审核项

**请求参数**:
```typescript
{
  "operation": "approve" | "reject" | "markNeedsInfo",  // 操作类型 (必填)
  "itemIds": string[],                   // 项目ID列表 (必填)
  "comment": string                      // 操作备注 (可选)
}
```

**响应**:
```typescript
{
  "code": 200,
  "message": "批量操作完成",
  "data": {
    "success": true,
    "processedCount": number             // 成功处理的记录数
  }
}
```

### 获取审核标签列表

**接口**: `GET /api/review-queue/tags`

**描述**: 获取系统预设的审核标签列表

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": [
    "会员",
    "积分规则",
    "支付问题",
    "退款",
    "产品"
    // 其他标签...
  ]
}
```

### 获取审核来源类型

**接口**: `GET /api/review-queue/sources`

**描述**: 获取系统支持的审核来源类型

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": [
    "用户提问-系统未答出",
    "用户反馈-答案差评",
    "低置信度回答"
    // 其他来源类型...
  ]
}
```

### 获取审核统计数据

**接口**: `GET /api/review-queue/stats`

**描述**: 获取审核队列统计信息

**响应**:
```typescript
{
  "code": 200,
  "message": "成功",
  "data": {
    "totalItems": number,         // 总审核项数
    "pendingCount": number,       // 待处理项目数
    "approvedCount": number,      // 已批准项目数
    "rejectedCount": number,      // 已拒绝项目数
    "needsInfoCount": number,     // 需要更多信息的项目数
    "bySource": {                 // 按来源的项目分布
      "用户提问-系统未答出": number,
      "用户反馈-答案差评": number,
      "低置信度回答": number
      // 其他来源...
    }
  }
}
```