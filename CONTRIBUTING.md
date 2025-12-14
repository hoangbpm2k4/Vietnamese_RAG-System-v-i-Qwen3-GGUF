# Contributing to Vietnamese RAG System

Cảm ơn bạn đã quan tâm đến dự án! 🎉

## How to Contribute

### 1. Report Bugs
- Mở issue mới với label `bug`
- Mô tả chi tiết: expected vs actual behavior
- Cung cấp code snippet để reproduce

### 2. Suggest Features
- Mở issue với label `enhancement`
- Giải thích use case và expected outcome

### 3. Submit Pull Requests

#### Setup Development Environment
```bash
# Fork và clone repo
git clone https://github.com/yourusername/vietnamese-rag-qwen3.git
cd vietnamese-rag-qwen3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Code Style
- Follow PEP 8
- Use type hints where possible
- Add docstrings for functions
- Keep functions focused and small

#### Before Submitting PR
- [ ] Code chạy được và pass tests
- [ ] Thêm comments cho logic phức tạp
- [ ] Update README nếu thay đổi API
- [ ] Format code với black (optional)

## Development Roadmap

### High Priority
- [ ] Vector database integration (Qdrant)
- [ ] FastAPI REST API
- [ ] Docker containerization

### Medium Priority
- [ ] LangChain/LangGraph integration
- [ ] Multi-document support
- [ ] Conversation memory

### Nice to Have
- [ ] Web UI với Streamlit/Gradio
- [ ] Monitoring dashboard
- [ ] A/B testing framework

## Questions?

- Open an issue

**Happy coding!** 🚀
