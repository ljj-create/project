# 数据库基础

## 关系型数据库

### ACID 特性
- **Atomicity（原子性）**: 事务要么全部成功，要么全部回滚
- **Consistency（一致性）**: 事务前后数据库保持一致状态
- **Isolation（隔离性）**: 并发事务互不干扰
- **Durability（持久性）**: 事务提交后数据永久保存

### 事务隔离级别
| 级别 | 脏读 | 不可重复读 | 幻读 |
|------|------|-----------|------|
| READ UNCOMMITTED | ✓ | ✓ | ✓ |
| READ COMMITTED | ✗ | ✓ | ✓ |
| REPEATABLE READ | ✗ | ✗ | ✓ |
| SERIALIZABLE | ✗ | ✗ | ✗ |

### 索引

#### B+ 树索引
- 叶子节点形成有序链表，支持范围查询
- 高度通常 3-4 层（千万级数据）
- 聚簇索引 vs 非聚簇索引

#### 索引优化
- **覆盖索引**: 查询字段都在索引中，无需回表
- **联合索引**: 遵循最左前缀原则
- **索引下推**: MySQL 5.6+，在存储引擎层过滤

### SQL 查询执行顺序
``FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT``

### JOIN 类型
- **INNER JOIN**: 两表交集
- **LEFT JOIN**: 左表全集 + 右表匹配
- **RIGHT JOIN**: 右表全集 + 左表匹配
- **FULL OUTER JOIN**: 两表并集

## 数据库范式
1. **1NF**: 每个字段不可再分
2. **2NF**: 满足 1NF，非主属性完全依赖于主键
3. **3NF**: 满足 2NF，非主属性不传递依赖于主键
4. **BCNF**: 每个决定因素都是候选键

## 事务实现原理

### MVCC（多版本并发控制）
- 每行数据保留多个版本
- 读操作读取快照，不加锁
- InnoDB 实现：隐藏字段（trx_id, roll_pointer）+ Undo Log

### WAL（Write-Ahead Logging）
- 先写日志，再写数据
- Redo Log 保证持久性
- Undo Log 保证原子性

## NoSQL

### Redis
- 内存数据库，支持多种数据结构
- String、List、Set、Hash、ZSet
- 应用：缓存、会话、排行榜、消息队列

### MongoDB
- 文档数据库，JSON 格式存储
- 灵活的 Schema
- 应用：日志、用户画像、内容管理

## 分布式数据库
- **CAP 定理**: 一致性、可用性、分区容错性最多满足两个
- **BASE**: 基本可用、软状态、最终一致性
- **分片策略**: 哈希分片、范围分片
- **一致性协议**: Paxos、Raft
