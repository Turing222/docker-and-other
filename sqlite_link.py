import sqlite3
import os

# 1. 定义数据库路径
DB_PATH = os.path.join("mlruns", "mlflow.db")

# 2. 连接数据库
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print(f"成功连接到数据库: {DB_PATH}")

    # 3. 示例查询：查看所有表格
    print("\n--- 所有数据表 ---")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(tables)

    # 4. 示例查询：获取最近 3 个 Runs 的 UUID 和状态
    print("\n--- 最近 3 个 Runs ---")
    query = """
    SELECT run_uuid, status, start_time FROM runs ORDER BY start_time DESC LIMIT 3;
    """
    cursor.execute(query)
    
    # 获取列名
    col_names = [description[0] for description in cursor.description]
    print(col_names)
    
    # 打印结果
    for row in cursor.fetchall():
        print(row)

finally:
    # 5. 关闭连接
    if 'conn' in locals() and conn:
        conn.close()