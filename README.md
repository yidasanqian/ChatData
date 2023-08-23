<div align="center">

# ChatData AI

ChatData AI 是一个基于 OpenAI 大语言模型的知识库问答系统，支持对本地文件进行问答。
</div>

## 💡 功能
- [x] 支持更便宜、更快的模型来完成问题的凝练工作，然后再使用昂贵的模型来回答问题
- [x] PDF文件上传
- [x] 本地会话
- [x] 流式输出
- [x] 支持部署在codesandbox

## 快速开始 :white_check_mark:  
要开始这个项目，您需要克隆存储库并在系统上安装 [Python](https://www.python.org/downloads/)
  
### 克隆存储库 :inbox_tray:
运行以下命令以克隆存储库:  

```
git clone https://github.com/yidasanqian/ChatData.git
```

### 安装依赖 :wrench: 
导航到项目目录:
```
cd ChatData
```

安装依赖项：
```
pip install -r requirements.txt
```

### 设置环境变量 ℹ️ 
新建`.env`文件：
```
OPENAI_API_KEY=
OPENAI_API_BASE=https://api.openai.com/v1
```

## 运行应用程序 :rocket:

若要运行应用程序，请运行以下命令：
```
flask --app server.app run
```
启用debug模式，请运行以下命令：
```
flask --app server.app --debug run
```

使用以下 URL 在浏览器中访问应用程序：
```
http://127.0.0.1:5000
```

## 使用协议
本仓库遵循 [Apache-2.0 license](https://github.com/yidasanqian/ChatData/blob/master/LICENSE) 开源协议。