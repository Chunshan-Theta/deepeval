# deepeval

**deepeval** 是一個用於處理與深度學習模型互動的工具集，旨在幫助開發者快速建立和測試相關 API 和工具。

## 目錄結構

```
deepeval/
│
├── examples_http_requests.py  # HTTP 請求示例
├── examples_ollama.py         # Ollama 示例
├── http_provider.py           # HTTP 提供程序邏輯
├── llmsetting.conf.example    # 配置範例
├── readme.md                  # 說明文件
├── requirements.txt           # 依賴列表
└── utils.py                   # 實用工具函數
```

## 功能

- 提供簡單的 HTTP 請求示例，便於學習和測試。
- 支援與 Ollama 工具的集成。
- 包含通用的 HTTP 提供程序和實用工具模組。

## 安裝

1. 克隆此專案到本地：
   ```bash
   git clone <repo-url>
   cd deepeval
   ```
2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 示例程式
運行示例程式來測試功能：
- HTTP 請求：
  ```bash
  python examples_http_requests.py
  ```
- Ollama 示例：
  ```bash
  python examples_ollama.py
  ```

### 配置
根據 `llmsetting.conf.example` 文件，設定自定義配置文件 `llmsetting.conf`。

## 貢獻指南

歡迎貢獻！請提交 Pull Request 或報告問題。

