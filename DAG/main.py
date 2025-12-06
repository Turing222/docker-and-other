import asyncio
import re
from typing import List

from sqlalchemy import String, select, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    async_sessionmaker,
    create_async_engine,
    AsyncSession
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

# 新增：NLP 相关的库
import jieba
from sentence_transformers import SentenceTransformer

# 1. 配置部分
DATABASE_URL = "postgresql+asyncpg://mlops_user:mlops_password@localhost:5432/mlops_db"
# 选择一个对中文支持很好的轻量级模型 (维度通常是 768)
MODEL_NAME = 'shibing624/text2vec-base-chinese' 

# 2. NLP 处理模块 (模拟 MLOps 中的模型服务)
class ChineseNLPProcessor:
    def __init__(self):
        print(f"⏳ Loading model '{MODEL_NAME}'... (might take a while first time)")
        # 加载预训练模型
        self.model = SentenceTransformer(MODEL_NAME)
        print("✅ Model loaded.")

    def clean_text(self, text: str) -> str:
        """基础预处理：去除特殊符号，保留中文、数字和英文"""
        # 使用正则表达式去除标点符号等
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)
        return text

    def get_embedding(self, text: str) -> List[float]:
        """将文本转换为向量"""
        # 1. 简单清洗
        cleaned_text = self.clean_text(text)
        # 2. (可选) 虽然 BERT 类模型不需要 jieba 分词，但在传统 NLP 中常用于关键词提取
        # 这里为了演示 jieba 的集成：
        words = jieba.lcut(cleaned_text) 
        print(f"   [Preprocess] Segments: {words}")
        
        # 3. 生成向量 (这是 CPU 密集型操作！)
        embedding = self.model.encode(cleaned_text)
        return embedding.tolist()

# 初始化全局 NLP 处理器
nlp_processor = None

# 3. 数据库模型定义
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

class Base(AsyncAttrs, DeclarativeBase):
    pass

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_content: Mapped[str] = mapped_column(String(1024)) # 原始文本
    
    # !注意!：text2vec-base-chinese 输出维度是 768
    embedding: Mapped[List[float]] = mapped_column(Vector(768)) 

    def __repr__(self):
        return f"<KB(id={self.id}, content='{self.raw_content[:20]}...')>"

# 4. 核心异步逻辑

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

async def add_document(session: AsyncSession, text_content: str):
    """
    异步添加文档：
    关键点：模型计算是 CPU 密集型的，不能直接阻塞异步循环。
    """
    print(f"\n➕ Adding: {text_content}")
    
    # !重要技巧!：使用 asyncio.to_thread 将 CPU 密集的向量化操作
    # 扔到另一个线程去跑，防止卡住整个程序的 Event Loop
    vector = await asyncio.to_thread(nlp_processor.get_embedding, text_content)
    
    doc = KnowledgeBase(raw_content=text_content, embedding=vector)
    session.add(doc)
    await session.commit()
    print("   ✅ Saved to DB.")

async def search_similar(session: AsyncSession, query_text: str, limit: int = 2):
    print(f"\n🔍 Query: '{query_text}'")
    
    # 1. 同样把查询文本的向量化过程放到线程池
    query_vec = await asyncio.to_thread(nlp_processor.get_embedding, query_text)
    
    # 2. 数据库查询 (IO 密集型，使用 await)
    stmt = select(KnowledgeBase).order_by(
        KnowledgeBase.embedding.l2_distance(query_vec)
    ).limit(limit)
    
    result = await session.execute(stmt)
    hits = result.scalars().all()
    
    print("   ⬇️ Results:")
    for hit in hits:
        # 计算距离通常也可以在 Python 算，但这里数据库已经排好序了
        print(f"   📄 {hit.raw_content}")

async def main():
    global nlp_processor
    # 在主程序开始时加载模型
    nlp_processor = ChineseNLPProcessor()
    
    await init_db()

    async with AsyncSessionLocal() as session:
        # 1. 准备一些中文语料
        corpus = [
            "机器学习是人工智能的一个子集，专注于利用数据进行训练。",
            "深度学习使用神经网络来模拟人脑的学习过程。",
            "Python是一种广泛使用的高级编程语言，非常适合数据科学。",
            "西红柿炒鸡蛋是一道非常受欢迎的中国家常菜。",
            "如何烹饪美味的牛排？需要控制好火候。",
        ]

        # 2. 插入数据
        for text in corpus:
            await add_document(session, text)
        
        # 3. 语义搜索测试
        # 案例 A: 搜技术相关
        await search_similar(session, "AI和神经网络的关系是什么？")
        
        # 案例 B: 搜食物相关
        await search_similar(session, "肚子饿了吃什么菜好？")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())