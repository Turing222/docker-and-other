import asyncio
import random
from typing import List

from sqlalchemy import String, select, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    async_sessionmaker,
    create_async_engine,
    AsyncSession
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector  # 导入 pgvector 类型

# 1. 数据库配置
# 注意使用 postgresql+asyncpg 协议
DATABASE_URL = "postgresql+asyncpg://mlops_user:mlops_password@localhost:5432/mlops_db"

# 创建异步引擎
engine = create_async_engine(DATABASE_URL, echo=True) # echo=True 会打印生成的 SQL，方便学习

# 创建异步 Session 工厂
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 2. 定义 ORM 模型
class Base(AsyncAttrs, DeclarativeBase):
    pass

class Document(Base):
    """
    模拟一个存储文档片段和向量的表
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(String(255))
    
    # 定义向量列，维度设为 3 (通常是 768, 1536 等，这里为了演示方便用 3)
    embedding: Mapped[List[float]] = mapped_column(Vector(3))

    def __repr__(self):
        return f"<Document(id={self.id}, content='{self.content}')>"

# 3. 核心功能函数

async def init_db():
    """初始化数据库：启用扩展并创建表"""
    async with engine.begin() as conn:
        # !重要!：必须先在数据库中启用 vector 扩展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # 删除旧表并重新创建（仅用于 Demo，生产环境请使用 Alembic 迁移）
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized and vector extension enabled.")

async def insert_data(session: AsyncSession):
    """插入一些模拟数据"""
    docs = [
        Document(content="Apple fruit", embedding=[1.0, 0.1, 0.0]),
        Document(content="Banana fruit", embedding=[0.9, 0.2, 0.0]),
        Document(content="Car vehicle", embedding=[0.0, 1.0, 0.2]),
        Document(content="Truck vehicle", embedding=[0.0, 0.9, 0.1]),
    ]
    session.add_all(docs)
    await session.commit()
    print(f"✅ Inserted {len(docs)} documents.")

async def vector_search(session: AsyncSession, query_vec: List[float], limit: int = 2):
    """
    执行向量相似度搜索
    """
    print(f"\n🔍 Searching for nearest neighbors to {query_vec}...")
    
    # 核心逻辑：使用 l2_distance (欧氏距离) 或 cosine_distance (余弦距离)
    # SQLAlchemy 2.0 语法
    stmt = select(Document).order_by(
        Document.embedding.l2_distance(query_vec)
    ).limit(limit)

    result = await session.execute(stmt)
    neighbors = result.scalars().all()

    for doc in neighbors:
        print(f"   -> Found: {doc.content} (ID: {doc.id})")

async def main():
    # 1. 初始化表结构
    await init_db()

    # 2. 数据操作
    async with AsyncSessionLocal() as session:
        # 插入数据
        await insert_data(session)
        
        # 查询案例 1: 找水果 (接近 [1, 0, 0])
        await vector_search(session, query_vec=[0.95, 0.05, 0.0])
        
        # 查询案例 2: 找车 (接近 [0, 1, 0])
        await vector_search(session, query_vec=[0.05, 0.95, 0.1])

    # 关闭引擎
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())