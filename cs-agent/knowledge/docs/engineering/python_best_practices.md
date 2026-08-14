# Python 工程实践

## 代码规范

### PEP 8
- 缩进：4 个空格
- 行长：≤ 79 字符（代码），≤ 72 字符（文档字符串）
- 命名：
  - 变量/函数：`snake_case`
  - 类：`PascalCase`
  - 常量：`UPPER_CASE`
  - 私有：`_leading_underscore`

### 类型提示（Type Hints）
```python
def greet(name: str, times: int = 1) -> list[str]:
    return [f"Hello, {name}!" for _ in range(times)]

# 复杂类型
from typing import Optional, Union
def process(data: list[dict[str, int]]) -> Optional[str]: ...
```

## 虚拟环境
```bash
# venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# conda
conda create -n myenv python=3.11
conda activate myenv

# 依赖管理
pip freeze > requirements.txt
pip install -r requirements.txt
```

## 项目结构
```
my_project/
├── README.md
├── pyproject.toml
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
└── docs/
```

## 测试

### pytest
```python
# test_core.py
import pytest

def test_add():
    assert add(1, 2) == 3

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add_parametrize(a, b, expected):
    assert add(a, b) == expected

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### 测试覆盖率
```bash
pip install pytest-cov
pytest --cov=my_package --cov-report=html
```

## 异步编程

### asyncio
```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.json()

async def main():
    # 并发执行
    results = await asyncio.gather(
        fetch_data("https://api.example.com/1"),
        fetch_data("https://api.example.com/2"),
    )

asyncio.run(main())
```

### 异步上下文管理器
```python
class AsyncDB:
    async def __aenter__(self):
        self.conn = await create_connection()
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
```

## 常用设计模式

### 单例模式
```python
class Singleton:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 工厂模式
```python
class LLMFactory:
    @staticmethod
    def create(provider: str, **kwargs):
        if provider == "openai":
            return OpenAILLM(**kwargs)
        elif provider == "qwen":
            return QwenLLM(**kwargs)
        raise ValueError(f"Unknown provider: {provider}")
```

### 策略模式
```python
class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy

    def sort(self, data):
        return self.strategy(data)

sorter = Sorter(strategy=sorted)
result = sorter.sort([3, 1, 2])
```

## 性能优化
1. **使用生成器**: 大数据集用 `yield` 而非返回列表
2. **避免全局变量**: 局部变量访问更快
3. **使用内置函数**: `map()`, `filter()`, `sum()` 等是 C 实现
4. **缓存**: `@functools.lru_cache` 缓存纯函数结果
5. **并行**: CPU 密集用 `multiprocessing`，IO 密集用 `asyncio`
