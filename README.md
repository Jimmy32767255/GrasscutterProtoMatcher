# GrasscutterProtoMatcher
# 割草机协议匹配器

## 简介

本工具用于比较两个 Protocol Buffers (.proto) 文件的结构相似度，包括消息(message)和枚举(enum)的匹配情况。

## 安装

1. 确保已安装 Python 3.7+
2. 安装依赖：
```
pip install -r requirements.txt
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| -h, --help | 显示帮助信息 |
| proto_file1 | 第一个 proto 文件路径 |
| proto_file2 | 第二个 proto 文件路径 |
| --hide-complete-match | 隐藏完全匹配的节点和边 |
| --log-to-file | 将日志输出到文件(log.txt) |
| --console-progress-only | 控制台只显示进度信息 |
| --insert-debug-delay | 在计算之间插入0.1s延迟(用于测试) |

### 特性

- 使用 RRWM 算法进行图匹配
- 使用匈牙利算法进行最优匹配
- 实时显示处理进度和CPU利用率

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE 文件](LICENSE)。
