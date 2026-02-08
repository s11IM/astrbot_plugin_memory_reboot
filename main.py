# ==============================================================================
# 第一部分：导入模块
# ==============================================================================

# ----- 1.1 Python 标准库 -----
import os
import re
import json
import time
import uuid
import hashlib
import datetime
import shutil
import gzip
from typing import Optional, List, Dict, Tuple

# ----- 1.2 第三方库（带依赖检查）-----
_missing_deps = []  # 记录缺失的依赖

try:
    import numpy as np  # 用于向量计算
except ImportError:
    _missing_deps.append("numpy")
    np = None

try:
    import aiohttp  # 用于异步HTTP请求
except ImportError:
    _missing_deps.append("aiohttp")
    aiohttp = None

HAS_PIL = False
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    # 虽然后备支持MD5，但强烈建议安装Pillow以获得dHash能力
    pass 

# ----- 1.3 AstrBot 框架 API -----
from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Plain, Image, Reply

# ----- 1.4 插件命令过滤相关 -----
try:
    from astrbot.core.star.filter.command import CommandFilter
    from astrbot.core.star.filter.command_group import CommandGroupFilter
    from astrbot.core.star.star_handler import star_handlers_registry, StarHandlerMetadata
    HAS_COMMAND_FILTER = True
except ImportError:
    HAS_COMMAND_FILTER = False
    logger.warning("[Memory Reboot] 无法导入命令过滤模块，自动过滤插件命令功能不可用")

# 启动时报告缺失依赖
if _missing_deps:
    logger.error(
        f"[Memory Reboot] 缺少依赖: {', '.join(_missing_deps)}，"
        f"请运行: pip install {' '.join(_missing_deps)}"
    )


# ==============================================================================
# 第二部分：常量定义
# ==============================================================================

DEFAULT_DATA_RETENTION_DAYS = 7           # 数据保留天数
DEFAULT_SIMILARITY_THRESHOLD = 0.95       # 文本相似度阈值
DEFAULT_IMAGE_HASH_THRESHOLD = 0.90       # 图片哈希相似度阈值（dHash 256位）
DEFAULT_MIN_UNIQUE_SENDERS = 3            # 最少不同发送者数量
DEFAULT_COOLDOWN_SECONDS = 3600           # 冷却时间（秒）
DEFAULT_MIN_TEXT_LENGTH = 2               # 最小文本长度
REMINDER_IMAGE_FILENAME = "1000101866.jpg" # 提醒图片文件名


# ==============================================================================
# 第三部分：主插件类
# ==============================================================================

class MemoryRebootPlugin(Star):
    """
    Memory Reboot - 记忆重现插件主类
    
    继承自 AstrBot 的 Star 基类，实现群消息监听和处理逻辑。
    
    主要功能模块:
        1. 消息监听与内容提取
        2. 文本Embedding相似度计算
        3. 图片感知哈希相似度计算
        4. LLM智能判断
        5. 提醒消息发送
    
    配置项说明（通过_conf_schema.json定义）:
        - blocked_groups: 黑名单群组列表
        - similarity_threshold: 文本相似度阈值 (0-1)
        - image_hash_threshold: 图片哈希相似度阈值 (0-1)
        - min_unique_senders: 触发提醒所需的最少不同发送者数量
        - cooldown_seconds: 同一话题的冷却时间（秒）
        - data_retention_days: 历史数据保留天数 (默认7天)
        - embedding_provider_id: Embedding模型提供商ID
        - vision_provider_id: 图片识别LLM的提供者ID
        - judge_provider_id: 判断LLM的提供者ID
    
    性能优化:
        - 内存缓存: 消息列表缓存在内存中，避免每次从磁盘加载
        - 增量写入: 新消息直接追加到缓存和磁盘，无需读-改-写
        - 懒加载: 仅在首次访问群组数据时从磁盘加载
    """
    
    # 内存缓存：{group_id: {"messages": [...], "last_load": timestamp, "dirty": bool}}
    _cache: Dict[str, Dict] = {}
    
    # ==========================================================================
    # 4.1 默认Prompt模板
    # ==========================================================================
    
    # 图片识别Prompt：用于判断图片类型并提取内容
    DEFAULT_VISION_PROMPT = """你是一个图像分类助手，你的任务是判断并分析图片里的内容。
最后直接返回JSON，不要输出其他内容。

## 核心判断逻辑

**is_sticker: true (需要被跳过的图片)**
*   **意图**：仅用于表达情绪、态度，或进行无意义的打招呼/冒泡。
*   **特征**：
    *   通用的互联网表情包（熊猫头、蘑菇头、各种猫猫、黄脸Emoji等）。
    *   单纯的人物/动漫/动物截图，配文仅为通用短语（如"收到"、"确实"、"急了"、"怎么会这样"）。
    *   没有文字内容或者文字内容缺乏具体指代，不包含专有名词、事件或独特观点。
    *   QQ/微信商店贴纸。

**is_sticker: false (不需要跳过的图片)**
*   **意图**：用于分享信息、发起话题、展示证据或讲述一个具体的"梗"。
*   **特征**：
    *   **截图类**：新闻快讯、文章正文、社交媒体帖子（微博/推特/小红书）、聊天记录、软件界面。
    *   **复杂梗图**：多格漫画、有前后文对比的图、或配文包含具体事件/行业痛点/复杂观点的梗图。
    *   **其他**：海报、公告、数据图表。
*   **原则**：如果不确定（例如文字较多的模糊图片），请标记为 false 以免漏掉潜在话题。

## Content 提取规则 (当 is_sticker=false 时)
务必详细提取，因为这些内容将用于生成搜索向量。
1.  **全面OCR**：提取图片中所有可辨识的文字，保留关键的数字、日期、ID。
2.  **场景描述**：简述图片类型（如"屏幕截图"、"文字较多的新闻"）。
3.  **视觉细节**：如果是梗图，描述画面发生了什么（如"左边是...右边是..."）。

**一致性要求**：
- 优先输出OCR文字原文，减少主观描述
- 使用固定格式：【类型】+ 客观内容
- 避免使用"可能"、"似乎"、"大概"等不确定词汇
- 不要添加个人解读或情感评价

## JSON示例

示例1（纯情绪表达 -> True）：
{"is_sticker": true, "content": "动漫角色流泪，配文'怎么会这样'"}

示例2（新闻/资讯截图 -> False）：
{"is_sticker": false, "content": "【新闻截图】来源：闪电新闻。标题：'月之暗面Kimi官方账号喊话百度'。正文提到百度搜索Kimi官网前4条全是广告。画面显示主持人吴阳欣蔚。"}

示例3（复杂/具体内容的梗图 -> False）：
{"is_sticker": false, "content": "【对比梗图】主题：Nostalgia be like。上图：2016年波奇酱在房间里流泪。下图：2026年波奇酱依然在流泪，脑子里想着2016年的自己。寓意：十年过去了没有任何改变。"}

请分析本图："""

    # 判断Prompt：用于LLM判断是否需要提醒用户
    DEFAULT_JUDGE_PROMPT = """你是一个社群消息分析员。判断当前消息是否属于"旧闻重发"，决定是否需要提醒用户。

## 已确认的前提
1. 当前消息与历史消息**检测到相似内容**
2. 已有 **{unique_count}人** 发送过类似内容（阈值：{min_senders}人）
3. 当前发送者**不是**最早的发送者

## 数据

**历史参考**（{matched_time_ago}）：
发送者：{matched_sender}
内容：{matched_content}

**历史上下文**：
{history_str}

**当前上下文**（最近消息）：
{current_str}

**待判断消息**：
发送者：{sender_name}
内容：{content}

## 判断逻辑

**提醒（should_remind: true）的情况：**
- 用户显然不知道群里之前讨论过这个话题
- 直接转发/分享内容，没有附带评论或讨论
- 单纯发一张图片（无上下文引用）
- 与当前讨论话题无关，突然出现的旧内容

**不提醒（should_remind: false）的情况：**
- 正在进行的讨论流中（最近几条消息都在讨论相关话题）。
- 已经开启了新话题，在新话题里并不是旧闻重发。
- 明确是对历史消息的回复、引用或补充
- 复读/接龙/玩梗行为
- 提供了新的信息增量（如后续进展、不同角度）
- **纯@消息**：消息仅包含@某用户名，无实质内容（如"@xxx"、"@xxx ?"），这类通常是社交互动而非信息分享
- **针对特定用户的短回复**：如"@xxx 收到"、"@xxx 好的"等日常交流

## 注意
- 很多用户只是单纯发图片或文字，没有"卧槽"之类的语气
- 判断重点是：**用户是否知道群里已经讨论过这件事**
- 如果当前上下文中没有相关讨论，且时间间隔较长，大概率是不知情的重发
- 但同时也注意是否已经开启了新的话题，用户的话在新话题里是否能与上下文产生联系

## 特殊情况处理
- **纯@消息不应触发提醒**：如果当前消息的主体内容仅仅是@某个用户名（可能带有少量语气词如"?"、"！"、"啊"等），这不是信息分享行为，而是社交互动，不应提醒
- **@+短语回复**：如"@xxx 收到"、"@xxx 确实"、"@xxx 哈哈"等属于日常社交互动，不应提醒
- **@+实质内容**：如"@xxx 这个新闻你看了吗：[内容]"才可能需要判断是否为旧闻重发

## 输出格式
简短分析后，输出JSON：

```json
{{
    "should_remind": true/false,
    "reason": "判断理由"
}}
```"""

    # ==========================================================================
    # 4.2 初始化方法
    # ==========================================================================
    
    def __init__(self, context: Context, config: dict):
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象，提供各种API访问能力
            config: 插件配置字典，来自_conf_schema.json的用户配置
        """
        super().__init__(context)
        self.config = config
        
        # 设置数据存储目录（支持旧版本数据自动迁移）
        self._setup_data_directory()
        
        # 插件所在目录（用于加载资源文件如提醒图片）
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化内存缓存
        self._cache = {}
        
        logger.info("[Memory Reboot] 插件初始化完成（含内存缓存优化）")
    
    def _setup_data_directory(self):
        """
        设置数据存储目录，并处理旧版本数据迁移
        
        迁移逻辑：
        - 路径1 (v0.x): data/old_news_reminder
        - 路径2 (v1.0+ 硬编码): data/memory_reboot (cwd下)
        - 路径3 (v1.2+ 标准): StarTools.get_data_dir()
        
        将旧路径数据迁移到标准路径。
        """
        # 1. 获取标准路径 (pathlib.Path)
        standard_path = StarTools.get_data_dir()
        
        # 2. 定义旧路径
        cwd = os.getcwd()
        legacy_v0 = os.path.join(cwd, "data", "old_news_reminder")
        legacy_v1 = os.path.join(cwd, "data", "memory_reboot")
        
        # 3. 检查是否需要迁移
        # 如果标准目录不存在，且存在旧数据，则尝试迁移
        if not standard_path.exists():
            target_source = None
            if os.path.exists(legacy_v1):
                target_source = legacy_v1
                logger.info(f"[Memory Reboot] 检测到 v1.0 旧数据目录: {legacy_v1}")
            elif os.path.exists(legacy_v0):
                target_source = legacy_v0
                logger.info(f"[Memory Reboot] 检测到 v0.x 旧数据目录: {legacy_v0}")
            
            if target_source:
                try:
                    # 确保标准路径的父目录存在
                    standard_path.parent.mkdir(parents=True, exist_ok=True)
                    # 移动目录
                    shutil.move(target_source, str(standard_path))
                    logger.info(f"[Memory Reboot] 数据成功迁移至标准路径: {standard_path}")
                except Exception as e:
                    logger.error(f"[Memory Reboot] 数据迁移失败: {e}")
                    # 迁移失败，回退使用旧路径以防数据丢失
                    self.data_dir = target_source
                    return

        # 4. 设置最终路径并确保存在
        self.data_dir = str(standard_path)
        os.makedirs(self.data_dir, exist_ok=True)
    
    # ==========================================================================
    # 4.3 配置检查方法
    # ==========================================================================
    
    def _is_group_enabled(self, group_id: str) -> bool:
        """
        检查指定群组是否启用了插件功能
        
        通过检查群组ID是否在黑名单中来判断。
        
        Args:
            group_id: 群组ID
            
        Returns:
            True: 群组已启用（不在黑名单中）
            False: 群组已禁用（在黑名单中）
        """
        blocked = self.config.get("blocked_groups", [])
        # 统一转换为字符串进行比较，避免类型不一致问题
        return str(group_id) not in [str(g) for g in blocked]
    
    # ==========================================================================
    # 4.4 插件命令过滤方法
    # ==========================================================================
    
    def _get_all_plugin_commands(self) -> List[str]:
        """
        获取所有已注册插件的命令列表
        
        通过遍历AstrBot的star_handlers_registry获取所有已注册的命令和命令组。
        这些命令将用于自动过滤，避免将其他插件的指令误判为重复内容。
        
        Returns:
            命令列表，例如 ["签到", "帮助", "天气", ...]
        """
        if not HAS_COMMAND_FILTER:
            logger.debug("[Memory Reboot] 命令过滤模块不可用，跳过获取插件命令")
            return []
        
        commands = set()  # 使用set去重
        
        try:
            # 获取所有插件的元数据
            all_stars_metadata = self.context.get_all_stars()
            all_stars_metadata = [star for star in all_stars_metadata if star.activated]
            
            if not all_stars_metadata:
                logger.debug("[Memory Reboot] 没有找到任何激活的插件")
                return []
            
            # 构建模块路径集合，用于匹配handler
            valid_module_paths = set()
            for star in all_stars_metadata:
                module_path = getattr(star, "module_path", None)
                if module_path:
                    valid_module_paths.add(module_path)
            
            # 遍历所有注册的处理器
            for handler in star_handlers_registry:
                if not isinstance(handler, StarHandlerMetadata):
                    continue
                
                # 检查此处理器是否属于已激活的插件
                if handler.handler_module_path not in valid_module_paths:
                    continue
                
                # 遍历处理器的过滤器，查找命令或命令组
                for filter_ in handler.event_filters:
                    if isinstance(filter_, CommandFilter):
                        # 添加主命令
                        if filter_.command_name:
                            commands.add(filter_.command_name)
                        # 添加别名
                        if hasattr(filter_, 'alias') and filter_.alias:
                            for alias in filter_.alias:
                                commands.add(alias)
                    elif isinstance(filter_, CommandGroupFilter):
                        # 添加命令组名称
                        if filter_.group_name:
                            commands.add(filter_.group_name)
            
            logger.debug(f"[Memory Reboot] 获取到 {len(commands)} 个插件命令")
            
        except Exception as e:
            logger.error(f"[Memory Reboot] 获取插件命令失败: {e}")
        
        return list(commands)
    
    def _build_command_patterns(self) -> List[str]:
        """
        根据插件命令构建正则表达式模式列表
        
        为每个命令生成匹配模式，支持以下格式：
        - 直接匹配命令（如 "签到"）
        - 带前缀的命令（如 "/签到", "!签到"）
        - 命令后带参数（如 "签到 xxx", "/天气 北京"）
        
        Returns:
            正则表达式模式列表
        """
        commands = self._get_all_plugin_commands()
        patterns = []
        
        for cmd in commands:
            if not cmd:
                continue
            # 转义正则特殊字符
            escaped_cmd = re.escape(cmd)
            # 匹配：可选前缀 + 命令 + 可选参数
            # 前缀包括: / ! # . 等常见命令前缀
            pattern = f"^[/!#\\.。]?{escaped_cmd}($|\\s.*$)"
            patterns.append(pattern)
        
        return patterns
    
    # 自己的命令白名单，这些命令不应该被过滤（需要由自己的命令处理器处理）
    SELF_COMMANDS = ["记忆状态", "查看过滤命令", "擦除记忆"]
    
    def _is_plugin_command(self, content: str) -> bool:
        """
        检查消息内容是否为其他插件的命令
        
        注意：自己的命令（SELF_COMMANDS）不会被过滤，以确保命令处理器能够正常工作。
        
        Args:
            content: 消息文本内容
            
        Returns:
            True: 是其他插件命令，应该被过滤
            False: 不是插件命令，或是自己的命令
        """
        if not self.config.get("auto_filter_commands", True):
            return False
        
        if not content:
            return False
        
        content_stripped = content.strip()
        
        # 检查是否是自己的命令（白名单），如果是则不过滤
        for self_cmd in self.SELF_COMMANDS:
            # 匹配：可选前缀 + 命令
            self_pattern = f"^[/!#\\.。]?{re.escape(self_cmd)}($|\\s.*$)"
            try:
                if re.match(self_pattern, content_stripped, re.IGNORECASE):
                    logger.debug(f"[Memory Reboot] 检测到自己的命令，不过滤: {content_stripped[:50]}")
                    return False
            except re.error:
                continue
        
        # 获取命令模式并检查是否是其他插件的命令
        patterns = self._build_command_patterns()
        
        for pattern in patterns:
            try:
                if re.match(pattern, content_stripped, re.IGNORECASE):
                    logger.debug(f"[Memory Reboot] 检测到插件命令: {content_stripped[:50]}")
                    return True
            except re.error:
                continue
        
        return False
    
    # ==========================================================================
    # 4.4 数据持久化方法 (分片存储 + Gzip压缩优化)
    # ==========================================================================
    
    def _get_group_dir(self, group_id: str) -> str:
        """获取群组专属数据目录"""
        path = os.path.join(self.data_dir, str(group_id))
        os.makedirs(path, exist_ok=True)
        return path

    def _get_daily_file(self, group_id: str, timestamp: float) -> str:
        """获取指定时间戳对应的日存储文件路径（.json.gz）"""
        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        return os.path.join(self._get_group_dir(group_id), f"{date_str}.json.gz")

    def _migrate_legacy_data(self, group_id: str):
        """将旧版单文件数据迁移到分片压缩存储"""
        legacy_path = os.path.join(self.data_dir, f"{group_id}_messages.json")
        if not os.path.exists(legacy_path):
            return
        
        try:
            logger.info(f"[Memory Reboot] 检测到旧数据，正在迁移群 {group_id} 到压缩分片存储...")
            with open(legacy_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
            
            # 按日期分组
            grouped = {}
            for msg in messages:
                ts = msg.get("timestamp", 0)
                date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                if date_str not in grouped:
                    grouped[date_str] = []
                grouped[date_str].append(msg)
            
            # 写入压缩分片
            group_dir = self._get_group_dir(group_id)
            for date_str, msgs in grouped.items():
                path = os.path.join(group_dir, f"{date_str}.json.gz")
                tmp_path = path + ".tmp"
                with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                    json.dump(msgs, f, ensure_ascii=False)
                os.replace(tmp_path, path)
            
            # 备份原文件
            os.rename(legacy_path, legacy_path + ".bak")
            logger.info(f"[Memory Reboot] 数据迁移完成！旧文件已备份为 .bak")
            
        except Exception as e:
            logger.error(f"[Memory Reboot] 数据迁移失败: {e}")

    def _load_messages(self, group_id: str) -> List[Dict]:
        """
        加载最近N天的消息（带内存缓存优化）
        
        优化策略:
        - 首次加载从磁盘读取，后续直接返回缓存
        - 每天首次访问时检查是否需要清理过期数据
        """
        group_id = str(group_id)
        
        # 检查缓存是否存在且有效
        if group_id in self._cache:
            cache_entry = self._cache[group_id]
            cache_date = datetime.date.fromtimestamp(cache_entry.get("last_load", 0))
            today = datetime.date.today()
            
            # 如果是同一天的缓存，直接返回
            if cache_date == today:
                return cache_entry["messages"]
            
            # 跨天了，需要清理过期消息并更新缓存
            retention_days = self.config.get("data_retention_days", 7)
            cutoff = time.time() - retention_days * 86400
            cache_entry["messages"] = [m for m in cache_entry["messages"] if m.get("timestamp", 0) > cutoff]
            cache_entry["last_load"] = time.time()
            return cache_entry["messages"]
        
        # 缓存不存在，从磁盘加载
        return self._load_messages_from_disk(group_id)
    
    def _load_messages_from_disk(self, group_id: str) -> List[Dict]:
        """从磁盘加载消息并更新缓存"""
        # 1. 尝试迁移单一大文件
        self._migrate_legacy_data(group_id)
        
        group_dir = self._get_group_dir(group_id)
        retention_days = self.config.get("data_retention_days", 7)
        
        # 2. 计算需要加载的日期范围
        today = datetime.date.today()
        valid_dates = { (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                        for i in range(retention_days + 1) }
        
        all_messages = []
        if os.path.exists(group_dir):
            try:
                for filename in sorted(os.listdir(group_dir)):
                    filepath = os.path.join(group_dir, filename)
                    
                    # 自动迁移：如果发现未压缩的 .json，压缩为 .json.gz
                    if filename.endswith(".json") and not filename.endswith(".json.gz"):
                        try:
                            # 读取未压缩数据
                            with open(filepath, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            # 写入压缩数据
                            gz_path = filepath + ".gz"
                            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False)
                            # 删除旧文件并更新路径
                            os.remove(filepath)
                            filename = filename + ".gz"
                            filepath = gz_path
                            logger.debug(f"[Memory Reboot] 已自动压缩文件: {filename}")
                        except Exception as e:
                            logger.error(f"[Memory Reboot] 自动压缩失败: {e}")
                            continue

                    # 加载 .json.gz 文件
                    if filename.endswith(".json.gz"):
                        date_part = filename.replace(".json.gz", "")
                        if date_part in valid_dates:
                            try:
                                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                                    day_msgs = json.load(f)
                                    all_messages.extend(day_msgs)
                            except Exception as e:
                                logger.error(f"[Memory Reboot] 读取压缩文件失败 {filepath}: {e}")
            except Exception as e:
                logger.error(f"[Memory Reboot] 读取消息失败: {e}")
                
        # 确保按时间排序
        all_messages.sort(key=lambda x: x.get("timestamp", 0))
        
        # 更新缓存
        self._cache[group_id] = {
            "messages": all_messages,
            "last_load": time.time()
        }
        
        logger.debug(f"[Memory Reboot] 从磁盘加载 {len(all_messages)} 条消息到缓存")
        return all_messages
    def _append_message(self, group_id: str, message: Dict):
        """
        追加单条消息（内存缓存优化版）
        
        优化策略:
        - 先更新内存缓存（O(1)操作）
        - 异步/延迟写入磁盘
        - 使用批量写入减少I/O次数
        """
        group_id = str(group_id)
        
        # 1. 更新内存缓存
        if group_id not in self._cache:
            self._cache[group_id] = {
                "messages": [],
                "last_load": time.time()
            }
        
        self._cache[group_id]["messages"].append(message)
        
        # 2. 写入磁盘（优化：每N条消息或跨天时才写入）
        cache_entry = self._cache[group_id]
        messages = cache_entry["messages"]
        
        # 获取当天的消息用于写入
        today_str = datetime.datetime.fromtimestamp(message.get("timestamp", time.time())).strftime("%Y-%m-%d")
        today_messages = [m for m in messages if datetime.datetime.fromtimestamp(m.get("timestamp", 0)).strftime("%Y-%m-%d") == today_str]
        
        # 每10条消息写入一次，或者是当天第一条消息时写入
        should_write = len(today_messages) == 1 or len(today_messages) % 10 == 0
        
        if should_write:
            self._write_daily_messages(group_id, today_str, today_messages)
    
    def _write_daily_messages(self, group_id: str, date_str: str, messages: List[Dict]):
        """将消息写入指定日期的文件"""
        path = os.path.join(self._get_group_dir(group_id), f"{date_str}.json.gz")
        tmp_path = path + ".tmp"
        
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"[Memory Reboot] 写入失败: {e}")
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
    
    def _flush_cache(self, group_id: str):
        """强制将缓存写入磁盘（用于关闭前或手动保存）"""
        group_id = str(group_id)
        if group_id not in self._cache:
            return
        
        messages = self._cache[group_id]["messages"]
        if not messages:
            return
        
        # 按日期分组写入
        grouped = {}
        for msg in messages:
            date_str = datetime.datetime.fromtimestamp(msg.get("timestamp", 0)).strftime("%Y-%m-%d")
            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append(msg)
        
        for date_str, day_msgs in grouped.items():
            self._write_daily_messages(group_id, date_str, day_msgs)
        
        logger.debug(f"[Memory Reboot] 已将群 {group_id} 的 {len(messages)} 条消息写入磁盘")
    
    def _cleanup_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        清理过期消息
        
        注意: 使用内存缓存后，_load_messages已经在跨天时自动清理过期消息。
        此方法保留以兼容现有调用，但通常直接返回原列表。
        """
        # 缓存模式下，过期清理已在 _load_messages 中完成
        # 这里只做简单的时间戳校验（通常不需要）
        if not messages:
            return messages
        
        now = time.time()
        retention_days = self.config.get("data_retention_days", 7)
        cutoff = now - retention_days * 86400
        
        # 快速检查：如果最老的消息都没过期，直接返回
        if messages[0].get("timestamp", 0) > cutoff:
            return messages
        
        return [m for m in messages if m.get("timestamp", 0) > cutoff]
    
    def _cleanup_image_cache(self, group_id: str):
        """
        清理过期的图片缓存文件
        
        图片缓存用于计算感知哈希。过期的缓存文件会被删除以节省磁盘空间。
        文件名格式: 20260202_041900_123456.jpg (日期_时间_微秒.jpg)
        
        Args:
            group_id: 群组ID
        """
        cache_dir = os.path.join(self.plugin_dir, "image_cache", group_id)
        
        if not os.path.exists(cache_dir):
            return
        
        retention_days = self.config.get("data_retention_days", 7)
        cutoff = time.time() - retention_days * 86400
        
        try:
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                
                # 从文件名解析时间戳
                try:
                    date_part = filename.split("_")[0]  # 20260202
                    time_part = filename.split("_")[1]  # 041900
                    dt = datetime.datetime.strptime(
                        f"{date_part}_{time_part}", 
                        "%Y%m%d_%H%M%S"
                    )
                    file_timestamp = dt.timestamp()
                    
                    # 删除过期文件
                    if file_timestamp < cutoff:
                        os.remove(filepath)
                        logger.debug(f"[Memory Reboot] 清理过期图片: {filename}")
                        
                except (ValueError, IndexError):
                    # 文件名格式不符合预期，跳过
                    pass
                    
        except Exception as e:
            logger.error(f"[Memory Reboot] 清理图片缓存失败: {e}")
    
    # ==========================================================================
    # 4.5 Embedding相关方法
    # ==========================================================================

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取文本的Embedding向量

        通过AstrBot provider调用embedding服务。
        优先使用 get_all_embedding_providers() 获取嵌入提供商。

        Args:
            text: 要编码的文本

        Returns:
            文本的向量表示（浮点数列表），失败返回None
        """
        provider_id = self.config.get("embedding_provider_id", "")
        if not provider_id:
            logger.debug("[Memory Reboot] Embedding: 未配置provider_id")
            return None

        try:
            provider = None

            # 优先从嵌入提供商列表中查找
            if hasattr(self.context, 'get_all_embedding_providers'):
                all_providers = self.context.get_all_embedding_providers()
                # 先尝试ID精确匹配
                for p in all_providers:
                    if hasattr(p, 'id') and p.id == provider_id:
                        provider = p
                        break
                # 如果ID匹配失败，尝试名称匹配
                if not provider:
                    for p in all_providers:
                        if hasattr(p, 'meta') and hasattr(p.meta, 'name'):
                            if p.meta.name == provider_id:
                                provider = p
                                break

            # 回退到通用方法
            if not provider:
                provider = self.context.get_provider_by_id(provider_id)

            if not provider:
                logger.debug(f"[Memory Reboot] 未找到嵌入提供商: {provider_id}")
                return None

            # 尝试多种嵌入方法
            methods = ['get_embeddings', 'embeddings', 'embedding', 'get_embedding']
            last_error = None
            for method_name in methods:
                if hasattr(provider, method_name):
                    method = getattr(provider, method_name)
                    try:
                        if method_name in ['get_embeddings', 'embeddings']:
                            result = await method([text])
                            if result and len(result) > 0:
                                return result[0]
                        else:
                            result = await method(text)
                            if result:
                                return result if isinstance(result, list) else list(result)
                    except Exception as e:
                        last_error = e
                        logger.debug(f"[Memory Reboot] 方法 {method_name} 调用失败: {e}")
                        continue

            if last_error:
                logger.warning(f"[Memory Reboot] 所有embedding方法均失败，最后错误: {last_error}")

        except Exception as e:
            logger.error(f"[Memory Reboot] 获取embedding失败: {e}")

        return None
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        余弦相似度公式: cos(θ) = (A·B) / (|A| × |B|)
        
        余弦相似度的特点:
        - 值域为 [-1, 1]
        - 1 表示向量方向完全相同（最相似）
        - 0 表示向量正交（无关）
        - -1 表示向量方向完全相反
        
        Args:
            v1: 向量1
            v2: 向量2
            
        Returns:
            相似度值，范围 [-1, 1]
        """
        # 参数校验
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        
        a, b = np.array(v1), np.array(v2)
        # 添加极小值1e-9避免除零错误
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    
    # ==========================================================================
    # 4.6 图片处理方法
    # ==========================================================================
    
    async def _cache_image(
        self, 
        url: str, 
        timestamp: float, 
        group_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        下载图片并缓存到本地，同时计算感知哈希
        
        Args:
            url: 图片的URL地址
            timestamp: 消息的时间戳（用于生成文件名）
            group_id: 群组ID（用于分目录存储）
            
        Returns:
            元组 (缓存文件路径, 图片哈希值)
            如果下载或处理失败，返回 (None, None)
        """
        try:
            # 创建群组专属的缓存目录
            cache_dir = os.path.join(self.plugin_dir, "image_cache", group_id)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 生成文件名: 20260202_041900_123456.jpg
            dt = datetime.datetime.fromtimestamp(timestamp)
            # 使用微秒部分确保文件名唯一
            filename = dt.strftime("%Y%m%d_%H%M%S") + f"_{int((timestamp % 1) * 1000000)}.jpg"
            filepath = os.path.join(cache_dir, filename)
            
            # 异步下载图片
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(filepath, "wb") as f:
                            f.write(await response.read())
                        logger.debug(f"[Memory Reboot] 图片缓存成功: {filename}")
                        
                        # 计算图片的感知哈希
                        image_hash = self._compute_image_hash(filepath)
                        if image_hash:
                            logger.debug(f"[Memory Reboot] 图片哈希: {image_hash}")
                        
                        return filepath, image_hash
                        
        except Exception as e:
            logger.error(f"[Memory Reboot] 图片缓存失败: {e}")
        
        return None, None
    
    def _compute_image_hash(self, image_path: str) -> Optional[str]:
        """
        计算图片的感知哈希值（Difference Hash / dHash算法）
        """
        if HAS_PIL:
            try:
                img = PILImage.open(image_path)
                # 使用17x16尺寸，宽度多1用于计算差值，生成16x16=256位哈希
                img = img.convert('L').resize((17, 16), PILImage.Resampling.LANCZOS)
                pixels = list(img.getdata())
                
                # dHash：比较相邻像素差异
                bits = []
                for row in range(16):
                    for col in range(16):
                        left_pixel = pixels[row * 17 + col]
                        right_pixel = pixels[row * 17 + col + 1]
                        bits.append('1' if left_pixel > right_pixel else '0')
                
                # 256位 = 64个十六进制字符
                hash_value = hex(int(''.join(bits), 2))[2:].zfill(64)
                return hash_value
            except Exception as e:
                logger.error(f"[Memory Reboot] 计算图片dHash失败: {e}")
                return None
        else:
            logger.debug("[Memory Reboot] PIL未安装，使用MD5哈希")
            return self._compute_file_md5(image_path)
    
    def _compute_file_md5(self, file_path: str) -> Optional[str]:
        """计算文件的MD5哈希（后备方案）"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"[Memory Reboot] 计算MD5失败: {e}")
            return None
    
    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """计算两个哈希值的相似度（基于汉明距离）"""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        try:
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            diff = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            return 1.0 - (diff / len(bin1))
        except Exception:
            return 0.0
    
    # ==========================================================================
    # 4.7 相似度匹配方法
    # ==========================================================================
    
    def _find_best_match(self, messages: List[Dict], embedding: List[float],
                         threshold: float, exclude_recent: int = 0) -> Tuple[Optional[Dict], int, float]:
        """查找最相似的历史消息（基于文本Embedding）"""
        best_sim, best_msg, best_idx = 0.0, None, -1
        search_range = len(messages) - exclude_recent if exclude_recent > 0 else len(messages)
        skipped_no_emb, skipped_dim_mismatch = 0, 0

        for i in range(search_range):
            msg = messages[i]
            emb = msg.get("embedding")
            if not emb:
                skipped_no_emb += 1
                continue
            if len(emb) != len(embedding):
                skipped_dim_mismatch += 1
                continue

            sim = self._cosine_similarity(embedding, emb)
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_msg, best_idx = msg, i

        # 只在有跳过的消息时记录
        if skipped_no_emb > 0 or skipped_dim_mismatch > 0:
            logger.debug(f"[Memory Reboot] 匹配统计: 跳过{skipped_no_emb}条无embedding, {skipped_dim_mismatch}条维度不匹配")

        return best_msg, best_idx, best_sim
    
    def _find_similar_image(self, messages: List[Dict], current_hash: str,
                            threshold: float = DEFAULT_IMAGE_HASH_THRESHOLD, exclude_recent: int = 0) -> Tuple[Optional[Dict], int, float]:
        """查找相似图片（基于图片哈希）"""
        best_sim, best_msg, best_idx = 0.0, None, -1
        search_range = len(messages) - exclude_recent if exclude_recent > 0 else len(messages)
        
        for i in range(search_range):
            msg = messages[i]
            msg_hash = msg.get("image_hash")
            if not msg_hash:
                continue
            sim = self._hash_similarity(current_hash, msg_hash)
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_msg, best_idx = msg, i
        return best_msg, best_idx, best_sim
    
    def _count_unique_senders(self, messages: List[Dict], embedding: List[float],
                               threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[int, List[str]]:
        """统计发送相似内容的不同用户数量"""
        sender_ids = set()
        for msg in messages:
            emb = msg.get("embedding")
            if emb and len(emb) == len(embedding) and self._cosine_similarity(embedding, emb) >= threshold:
                if msg.get("sender_id"):
                    sender_ids.add(msg.get("sender_id"))
        return len(sender_ids), list(sender_ids)
    
    def _count_unique_senders_by_hash(self, messages: List[Dict], image_hash: str,
                                       threshold: float = DEFAULT_IMAGE_HASH_THRESHOLD) -> Tuple[int, List[str]]:
        """统计发送相似图片的不同用户数量"""
        sender_ids = set()
        for msg in messages:
            msg_hash = msg.get("image_hash")
            if msg_hash and self._hash_similarity(image_hash, msg_hash) >= threshold:
                if msg.get("sender_id"):
                    sender_ids.add(msg.get("sender_id"))
        return len(sender_ids), list(sender_ids)
    
    def _get_context_around(self, messages: List[Dict], index: int, before: int = 40, after: int = 40) -> List[Dict]:
        """获取指定消息前后的上下文"""
        start = max(0, index - before)
        end = min(len(messages), index + after + 1)
        return [{"sender_name": m.get("sender_name"), "content": m.get("content"), "timestamp": m.get("timestamp")} 
                for m in messages[start:end]]
    
    # ==========================================================================
    # 4.8 时间格式化方法
    # ==========================================================================
    
    def _format_time(self, timestamp: float) -> str:
        """格式化时间戳为 MM-DD HH:MM:SS"""
        return datetime.datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M:%S")
    
    def _format_time_ago(self, timestamp: float) -> str:
        """格式化为"多久之前"的形式"""
        diff = time.time() - timestamp
        if diff < 60:
            return f"{int(diff)}秒前"
        elif diff < 3600:
            return f"{int(diff / 60)}分钟前"
        elif diff < 86400:
            return f"{int(diff / 3600)}小时前"
        else:
            return f"{int(diff / 86400)}天前"
    
    # ==========================================================================
    # 4.9 LLM交互方法
    # ==========================================================================
    
    async def _image_to_text(self, url: str) -> Optional[Tuple[str, str]]:
        """使用LLM识别图片内容，返回(内容, 类型)或None（表情包）"""
        provider_id = self.config.get("vision_provider_id", "")
        if not provider_id:
            logger.debug("[Memory Reboot] 图片识别: 未配置vision_provider_id")
            return None
        
        try:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.debug(f"[Memory Reboot] 图片识别: 未找到provider={provider_id}")
                return None
            
            logger.debug(f"[Memory Reboot] 图片识别: 开始调用视觉模型...")
            
            prompt = self.config.get("vision_prompt") or self.DEFAULT_VISION_PROMPT
            response = await provider.text_chat(prompt=prompt, contexts=[], image_urls=[url])
            
            if response and response.completion_text:
                result = response.completion_text.strip()
                
                try:
                    text = result
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                    
                    parsed = json.loads(text.strip())
                    raw_sticker = parsed.get("is_sticker", False)
                    if isinstance(raw_sticker, str):
                        is_sticker = raw_sticker.lower() in ("true", "yes", "1", "t")
                    else:
                        is_sticker = bool(raw_sticker)
                    content = parsed.get("content", "")
                    
                    if "type" in parsed and parsed.get("type", "").lower() == "emoji":
                        is_sticker = True
                    
                    if is_sticker:
                        return None
                    
                    return (content, "content")
                    
                except json.JSONDecodeError:
                    sticker_keywords = ["is_sticker\": true", "表情包", "emoji", "sticker", "熊猫头", "滑稽"]
                    if any(kw in result.lower() for kw in sticker_keywords):
                        return None
                    return (result, "unknown")
        except Exception as e:
            logger.error(f"[Memory Reboot] 图片识别失败: {e}")
        return None
    
    async def _judge_remind(self, content: str, sender_name: str, matched_msg: Dict,
                            history_ctx: List[Dict], current_ctx: List[Dict], unique_count: int) -> bool:
        """LLM判断是否需要提醒"""
        provider_id = self.config.get("judge_provider_id", "")
        if not provider_id:
            logger.debug("[Memory Reboot] LLM判断: 未配置judge_provider_id，跳过判断")
            return False
        
        try:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.debug(f"[Memory Reboot] LLM判断: 未找到provider={provider_id}")
                return False
            
            def fmt(m):
                ts = m.get('timestamp', 0)
                time_str = self._format_time(ts) if ts else "??:??:??"
                return f"[{time_str}] {m.get('sender_name', '?')}: {m.get('content', '')}"
            
            history_str = "\n".join([fmt(m) for m in history_ctx])
            current_str = "\n".join([fmt(m) for m in current_ctx])
            
            matched_time_ago = self._format_time_ago(matched_msg.get("timestamp", 0))
            matched_sender = matched_msg.get("sender_name", "未知")
            matched_content = matched_msg.get("content", "")[:300]
            min_senders = self.config.get("min_unique_senders", 3)
            
            prompt_template = self.config.get("judge_prompt") or self.DEFAULT_JUDGE_PROMPT
            
            try:
                prompt = prompt_template.format(
                    matched_time=self._format_time(matched_msg.get("timestamp", 0)),
                    matched_time_ago=matched_time_ago, matched_sender=matched_sender,
                    matched_content=matched_content, history_str=history_str,
                    current_str=current_str, sender_name=sender_name,
                    content=content, min_senders=min_senders,
                    unique_count=unique_count
                )
            except Exception:
                # 兼容旧版提示词如果不包含 {unique_count} 的情况
                prompt = self.DEFAULT_JUDGE_PROMPT.format(
                    matched_time=self._format_time(matched_msg.get("timestamp", 0)),
                    matched_time_ago=matched_time_ago, matched_sender=matched_sender,
                    matched_content=matched_content, history_str=history_str,
                    current_str=current_str, sender_name=sender_name,
                    content=content, min_senders=min_senders,
                    unique_count=unique_count
                )

            response = await provider.text_chat(prompt=prompt, contexts=[])
            
            if response and response.completion_text:
                text = response.completion_text
                if "```" in text:
                    text = text.split("```json")[-1].split("```")[0] if "```json" in text else text.split("```")[1].split("```")[0]
                try:
                    result = json.loads(text.strip())
                    should_remind = result.get("should_remind", False)
                    reason = result.get('reason', '无')
                    logger.debug(f"[Memory Reboot] LLM返回: should_remind={should_remind}")
                    logger.debug(f"[Memory Reboot] LLM理由: {reason}")
                    return should_remind
                except Exception as e:
                    logger.debug(f"[Memory Reboot] LLM返回JSON解析失败: {e}, 尝试文本匹配")
                    matched = '"should_remind": true' in text.lower()
                    logger.debug(f"[Memory Reboot] 文本匹配结果: {matched}")
                    return matched
        except Exception as e:
            logger.error(f"[Memory Reboot] LLM判断异常: {e}")
        return False
    
    async def _send_reminder(self, event: AstrMessageEvent):
        """发送提醒消息（使用引用回复形式）"""
        # 尝试获取消息ID用于引用回复
        msg_id = None
        try:
            msg_id = getattr(event.message_obj, "message_id", None)
        except Exception:
            msg_id = None
        
        chain = []
        # 使用引用回复
        if msg_id:
            chain.append(Reply(id=str(msg_id)))
        
        img_path = os.path.join(self.plugin_dir, REMINDER_IMAGE_FILENAME)
        if os.path.exists(img_path):
            chain.append(Image.fromFileSystem(img_path))
        else:
            chain.append(Plain("这个话题之前已经有人讨论过了哦~"))
        yield event.chain_result(chain)
    
    async def _extract_content(self, event: AstrMessageEvent) -> Optional[Tuple[str, Optional[str]]]:
        """提取消息内容，返回(文本内容, 图片URL)"""
        text = event.message_str.strip() if event.message_str else ""
        urls = []
        if hasattr(event, "message_obj") and event.message_obj:
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    url = getattr(comp, "url", None) or getattr(comp, "file", None)
                    if url:
                        urls.append(url)
        
        if urls:
            for url in urls:
                result = await self._image_to_text(url)
                if result:
                    img_text, _ = result
                    content = f"{text} [图片内容: {img_text}]" if text else f"[图片内容: {img_text}]"
                    logger.debug(f"[Memory Reboot] 图片转文本成功")
                    return (content, url)
                else:
                    logger.info(f"[Memory Reboot] 跳过表情包")
            if not text:
                logger.debug(f"[Memory Reboot] 无有效内容，跳过")
                return None
        return (text, None) if text else None
    
    # ==========================================================================
    # 4.10 主消息处理器
    # ==========================================================================
    
    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息的主入口"""
        group_id = event.get_group_id()
        
        # 群组检查
        if not group_id:
            logger.debug("[Memory Reboot] 跳过: 非群组消息")
            return
        if not self._is_group_enabled(group_id):
            logger.debug(f"[Memory Reboot] 跳过: 群{group_id}在黑名单中")
            return
        
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or sender_id
        logger.debug(f"[Memory Reboot] ━━━ 收到消息 ━━━ 群:{group_id} 发送者:{sender_name}({sender_id})")
        
        result = await self._extract_content(event)
        if not result:
            logger.debug(f"[Memory Reboot] 跳过: 内容提取失败或为表情包")
            return
        
        content, image_url = result

        # 过滤：最小长度检查
        if not image_url:
            min_length = self.config.get("min_text_length", DEFAULT_MIN_TEXT_LENGTH)
            if len(content) < min_length:
                logger.debug(f"[Memory Reboot] 跳过: 短文本({len(content)}<{min_length})")
                return

        # 过滤：正则表达式检查
        for pattern in self.config.get("ignore_regex", []):
            try:
                # 使用 fullmatch 确保完全匹配，避免误伤包含关键词的普通句子
                # 例如：pattern="何意味" 时
                # fullmatch: 匹配 "何意味"，不匹配 "你这是何意味啊"
                # search: 两者都匹配
                if re.fullmatch(pattern, content):
                    logger.info(f"[Memory Reboot] 已正则匹配: {pattern}")
                    return
            except re.error:
                pass

        # 过滤：插件命令自动过滤
        if self._is_plugin_command(content):
            logger.info(f"[Memory Reboot] 跳过: 检测到插件命令")
            return

        # 加载消息（使用内存缓存）
        messages = self._load_messages(group_id)
        logger.debug(f"[Memory Reboot] 历史消息: {len(messages)}条（缓存）")
        
        # 定期清理图片缓存（每100条消息触发一次）
        if len(messages) % 100 == 0 and len(messages) > 0:
            self._cleanup_image_cache(group_id)
        
        # 生成embedding和图片哈希
        embedding = await self._get_embedding(content)
        logger.debug(f"[Memory Reboot] Embedding: {'成功获取' if embedding else '获取失败'}, 维度={len(embedding) if embedding else 0}")
        now = time.time()
        cached_image, image_hash = None, None
        if image_url:
            cached_image, image_hash = await self._cache_image(image_url, now, group_id)
            logger.debug(f"[Memory Reboot] 图片缓存: {'成功' if cached_image else '失败'}, 哈希={'有' if image_hash else '无'}")
        
        # 创建当前消息记录
        msg = {
            "id": str(uuid.uuid4()), "sender_id": sender_id, "sender_name": sender_name,
            "content": content, "timestamp": now, "embedding": embedding,
            "has_image": image_url is not None, "cached_image": cached_image, "image_hash": image_hash
        }
        
        # 注意：此时不追加到 messages 列表，而是创建一个包含当前消息的临时列表用于匹配
        # 实际的追加会在 _append_message 中完成
        messages_with_current = messages + [msg]
        
        # 相似度匹配（使用包含当前消息的列表，但排除最后一条）
        matched_msg, matched_idx, match_type = None, -1, None
        text_threshold = self.config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        image_hash_threshold = self.config.get("image_hash_threshold", DEFAULT_IMAGE_HASH_THRESHOLD)
        
        if embedding:
            emb_matched, emb_idx, emb_sim = self._find_best_match(messages_with_current, embedding, text_threshold, exclude_recent=1)
            logger.info(f"[Memory Reboot] 文本相似度: {emb_sim:.4f} (阈值{text_threshold})")
            if emb_matched:
                matched_msg, matched_idx, match_type = emb_matched, emb_idx, "embedding"

        if image_hash:
            img_matched, img_idx, img_sim = self._find_similar_image(messages_with_current, image_hash, image_hash_threshold, exclude_recent=1)
            logger.info(f"[Memory Reboot] 图片哈希相似度: {img_sim:.4f} (阈值{image_hash_threshold})")
            if not matched_msg and img_matched:
                matched_msg, matched_idx, match_type = img_matched, img_idx, "image_hash"
            elif matched_msg and img_matched and img_matched.get("timestamp", 0) < matched_msg.get("timestamp", 0):
                matched_msg, matched_idx, match_type = img_matched, img_idx, "image_hash"
        
        if not matched_msg:
            logger.debug(f"[Memory Reboot] 未找到匹配消息，仅保存记录")
            self._append_message(group_id, msg)
            return
        
        # 人数检测（使用包含当前消息的列表）
        min_unique_senders = self.config.get("min_unique_senders", 3)
        if match_type == "embedding" and embedding:
            unique_count, sender_list = self._count_unique_senders(messages_with_current, embedding, text_threshold)
            logger.debug(f"[Memory Reboot] 人数检测(文本): {unique_count}人发送过相似内容")
        elif match_type == "image_hash" and image_hash:
            unique_count, sender_list = self._count_unique_senders_by_hash(messages_with_current, image_hash, image_hash_threshold)
            logger.debug(f"[Memory Reboot] 人数检测(图片): {unique_count}人发送过相似图片")
        else:
            unique_count = 1
            logger.debug(f"[Memory Reboot] 人数检测: 无法统计(embedding或hash缺失)")
        
        if sender_id == matched_msg.get("sender_id"):
            logger.debug(f"[Memory Reboot] ✗ 跳过: 同一用户({sender_name})重发自己的内容")
            self._append_message(group_id, msg)
            return
        
        if unique_count < min_unique_senders:
            logger.debug(f"[Memory Reboot] ✗ 跳过: 不同用户数{unique_count}<{min_unique_senders}(阈值)")
            self._append_message(group_id, msg)
            return
        
        logger.debug(f"[Memory Reboot] ✓ 通过人数检测: {unique_count}人>={min_unique_senders}人")
        
        # 冷却时间检查
        time_diff = now - matched_msg.get("timestamp", 0)
        cooldown = self.config.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)
        if time_diff < cooldown:
            logger.debug(f"[Memory Reboot] ✗ 跳过: 冷却时间内({int(time_diff)}s<{cooldown}s)")
            self._append_message(group_id, msg)
            return
        
        logger.debug(f"[Memory Reboot] ✓ 通过冷却检测: 间隔{int(time_diff)}s>={cooldown}s")
        
        # 检查是否启用LLM判断
        enable_llm_judge = self.config.get("enable_llm_judge", True)
        
        if enable_llm_judge:
            # LLM判断
            history_ctx = self._get_context_around(messages_with_current, matched_idx, before=40, after=40)
            current_ctx = [{"sender_name": m.get("sender_name"), "content": m.get("content"), "timestamp": m.get("timestamp")}
                           for m in messages_with_current[:-1][-40:]]

            logger.debug(f"[Memory Reboot] 进入LLM判断: 匹配={match_type}, 来自={matched_msg.get('sender_name')}, {self._format_time_ago(matched_msg.get('timestamp', 0))}")

            should_remind = await self._judge_remind(content, sender_name, matched_msg, history_ctx, current_ctx, unique_count)
        else:
            # 关闭LLM判断时，直接触发提醒
            logger.debug(f"[Memory Reboot] LLM判断已关闭，直接触发提醒")
            should_remind = True

        self._append_message(group_id, msg)

        if should_remind:
            logger.info(f"[Memory Reboot] 最终判断: 触发提醒 -> {sender_name}")
            async for result in self._send_reminder(event):
                yield result
        else:
            logger.info(f"[Memory Reboot] 最终判断: 不提醒")
    
    # ==========================================================================
    # 4.11 命令处理器
    # ==========================================================================
    
    @filter.command("记忆状态")
    async def check_status(self, event: AstrMessageEvent):
        """查看插件状态（仅管理员）"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("请在群聊中使用")
            return
        if not event.is_admin():
            yield event.plain_result("❌ 仅管理员可执行")
            return
        
        messages = self._load_messages(group_id)
        img_path = os.path.join(self.plugin_dir, REMINDER_IMAGE_FILENAME)
        
        # 获取黑名单状态
        is_blocked = not self._is_group_enabled(group_id)
        blocked_status = "🚫 已拉黑 (不记录/不提醒)" if is_blocked else "✅ 正常工作"

        # 获取插件命令过滤状态
        auto_filter_enabled = self.config.get('auto_filter_commands', True)
        if auto_filter_enabled and HAS_COMMAND_FILTER:
            plugin_commands = self._get_all_plugin_commands()
            cmd_filter_status = f"✅ 已启用 (检测到{len(plugin_commands)}个命令)"
        elif auto_filter_enabled and not HAS_COMMAND_FILTER:
            cmd_filter_status = "⚠️ 已启用但模块不可用"
        else:
            cmd_filter_status = "❌ 已禁用"
        
        # 获取LLM判断开关状态
        enable_llm_judge = self.config.get('enable_llm_judge', True)
        judge_provider_id = self.config.get('judge_provider_id', '')
        if enable_llm_judge:
            if judge_provider_id:
                llm_judge_status = f"✅ 已启用 (提供商: {judge_provider_id})"
            else:
                llm_judge_status = "⚠️ 已启用但未配置提供商"
        else:
            llm_judge_status = "❌ 已禁用 (匹配即提醒)"
        
        status = f"""✅ Memory Reboot - 记忆状态

📌 群号: {group_id}
📊 消息数: {len(messages)}
🧠 含embedding: {sum(1 for m in messages if m.get("embedding"))}
🖼️ 含图片: {sum(1 for m in messages if m.get("has_image"))} (含哈希: {sum(1 for m in messages if m.get("image_hash"))})

⚙️ 配置参数:
📏 文本相似度阈值: {self.config.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)}
🔍 图片哈希阈值: {self.config.get('image_hash_threshold', DEFAULT_IMAGE_HASH_THRESHOLD)}
👥 最少不同用户: {self.config.get('min_unique_senders', DEFAULT_MIN_UNIQUE_SENDERS)}人
⏰ 冷却时间: {self.config.get('cooldown_seconds', DEFAULT_COOLDOWN_SECONDS)}秒
📅 数据保留: {self.config.get('data_retention_days', DEFAULT_DATA_RETENTION_DAYS)}天

🛠️ 环境检查:
- Pillow库: {'✅ 已安装 (dHash可用)' if HAS_PIL else '❌ 未安装 (降级为MD5)'}
- 提醒图片: {'✅ 存在' if os.path.exists(img_path) else '⚠️ 不存在 (将发送纯文本)'}
- 命令过滤: {cmd_filter_status}
- LLM判断: {llm_judge_status}"""
        yield event.plain_result(status)
    
    @filter.command("查看过滤命令")
    async def show_filtered_commands(self, event: AstrMessageEvent):
        """查看当前检测到的所有插件命令（仅管理员）"""
        if not event.is_admin():
            yield event.plain_result("❌ 仅管理员可执行")
            return
        if not HAS_COMMAND_FILTER:
            yield event.plain_result("❌ 命令过滤模块不可用，请检查AstrBot版本")
            return
        
        commands = self._get_all_plugin_commands()
        
        if not commands:
            yield event.plain_result("📋 当前未检测到任何插件命令")
            return
        
        # 将命令分组显示，每行最多5个
        cmd_lines = []
        for i in range(0, len(commands), 5):
            cmd_lines.append("  ".join(commands[i:i+5]))
        
        auto_filter_enabled = self.config.get('auto_filter_commands', True)
        status = "✅ 已启用" if auto_filter_enabled else "❌ 已禁用"
        
        result = f"""📋 插件命令过滤列表

🔧 自动过滤状态: {status}
📊 检测到 {len(commands)} 个命令:

{chr(10).join(cmd_lines)}

💡 这些命令会被自动忽略，不会被记录或触发旧闻提醒。
可在配置中修改 "auto_filter_commands" 来启用/禁用此功能。"""
        
        yield event.plain_result(result)
    
    @filter.command("擦除记忆")
    async def clear_data(self, event: AstrMessageEvent):
        """清除群组数据（仅管理员，结果仅输出到日志）"""
        group_id = event.get_group_id()
        if not group_id:
            logger.info("[Memory Reboot] 擦除记忆命令失败: 非群聊环境")
            return
        if not event.is_admin():
            logger.info(f"[Memory Reboot] 擦除记忆命令被拒绝: 用户 {event.get_sender_id()} 非管理员")
            return
        
        group_id = str(group_id)
        
        # 清除内存缓存
        if group_id in self._cache:
            del self._cache[group_id]
        
        # 删除整个群组数据目录
        group_dir = self._get_group_dir(group_id)
        if os.path.exists(group_dir):
            try:
                shutil.rmtree(group_dir)
                logger.info(f"[Memory Reboot] 记忆已擦除 - 群{group_id}的所有历史数据已清空（含内存缓存）")
            except Exception as e:
                logger.error(f"[Memory Reboot] 清空数据失败: {e}")
        else:
            logger.info(f"[Memory Reboot] 记忆擦除完成 - 群{group_id}暂无数据")
