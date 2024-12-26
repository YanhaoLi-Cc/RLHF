import logging
import sys

from typing import Optional, Tuple, Union, List, Dict, Any


# 定义日志格式常量
_FORMAT: str = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
"""
日志消息的格式字符串。包含以下字段:
- levelname: 日志级别名称
- asctime: 时间戳
- filename: 源文件名
- lineno: 行号
- message: 日志消息
"""

_DATE_FORMAT: str = "%m-%d %H:%M:%S"
"""
日期时间格式字符串。格式为: MM-DD HH:MM:SS
"""

class NewLineFormatter(logging.Formatter):
    """
    自定义日志格式化器，用于处理多行日志消息的格式化。
    
    继承自logging.Formatter，主要增强了对包含换行符消息的处理，
    确保多行消息的每一行都保持相同的前缀格式。

    Attributes:
        fmt (str): 日志格式字符串
        datefmt (str): 日期时间格式字符串

    Example:
        >>> # 创建formatter实例
        >>> formatter = NewLineFormatter(
        ...     fmt=_FORMAT,
        ...     datefmt=_DATE_FORMAT
        ... )
        >>> 
        >>> # 配置logger
        >>> logger = logging.getLogger('app')
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)
        >>> logger.addHandler(handler)
        >>> 
        >>> # 记录多行日志
        >>> logger.info("Line 1\\nLine 2\\nLine 3")
        INFO 12-23 14:30:45 test.py:10] Line 1
        INFO 12-23 14:30:45 test.py:10] Line 2
        INFO 12-23 14:30:45 test.py:10] Line 3
    """

    def __init__(self, fmt: str, datefmt: Optional[str] = None):
        """
        初始化格式化器。

        Args:
            fmt: 日志格式字符串
            datefmt: 日期时间格式字符串，可选

        Note:
            - 继承基类初始化
            - 支持自定义日期格式
            - 保持与标准Formatter兼容
        """
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录。

        对日志记录进行格式化，特别处理包含换行符的消息，
        确保每行都包含相同的前缀信息。

        Args:
            record: 日志记录对象

        Returns:
            str: 格式化后的日志字符串

        Note:
            - 处理多行消息
            - 保持格式一致性
            - 支持所有日志级别
        """
        # 使用基类格式化
        msg = super().format(record)
        
        # 检查消息非空
        if record.message != "":
            # 分割消息获取前缀
            parts = msg.split(record.message)
            # 替换换行符，添加前缀
            msg = msg.replace("\n", "\r\n" + parts[0])
            
        return msg
    
# 定义全局logger和handler
_root_logger: logging.Logger = logging.getLogger("rlhf")
"""根日志记录器,用于RLHF(强化学习人类反馈)模块的日志记录"""

_default_handler: Optional[logging.StreamHandler] = None
"""默认的日志处理器,使用标准输出流"""

def _setup_logger() -> None:
    """
    配置RLHF模块的日志系统。
    
    设置根日志记录器的配置,包括日志级别、处理器、格式化器等。
    该函数实现了以下功能:
    1. 设置根日志器的日志级别为DEBUG
    2. 创建并配置标准输出流处理器
    3. 设置自定义的日志格式
    4. 禁用日志向上传播

    Global Variables Modified:
        _root_logger: 根日志记录器
        _default_handler: 默认日志处理器

    Implementation Details:
        1. 日志级别配置:
           - 根记录器: DEBUG级别(捕获所有日志)
           - 处理器: INFO级别(只输出INFO及以上级别)

        2. 处理器配置:
           - 使用sys.stdout作为输出流
           - 继承stdout的flush行为
           - 确保单例模式

        3. 格式化器设置:
           - 使用NewLineFormatter
           - 应用预定义的格式和日期格式

        4. 传播控制:
           - 禁用向父记录器传播
           - 避免重复日志

    Example:
        >>> # 基本使用
        >>> _setup_logger()
        >>> _root_logger.info("Starting RLHF training...")
        INFO 12-23 14:30:45 train.py:10] Starting RLHF training...
        
        >>> # 多行日志
        >>> _root_logger.debug("Debug info\\nLine 2")
        DEBUG 12-23 14:30:45 train.py:11] Debug info
        DEBUG 12-23 14:30:45 train.py:11] Line 2

    Note:
        - 这是一个内部函数,通常不需要直接调用
        - 会自动处理handler的单例模式
        - 支持多行日志格式化
        - 线程安全考虑
    """
    # 设置根日志器级别为DEBUG(捕获所有级别日志)
    _root_logger.setLevel(logging.DEBUG)
    
    # 使用全局handler确保单例模式
    global _default_handler
    
    # 如果handler未初始化,创建并配置新handler
    if _default_handler is None:
        # 创建标准输出流handler
        _default_handler = logging.StreamHandler(sys.stdout)
        # 继承stdout的flush行为
        _default_handler.flush = sys.stdout.flush  # type: ignore
        # 设置handler的输出级别为INFO
        _default_handler.setLevel(logging.INFO)
        # 将handler添加到根日志器
        _root_logger.addHandler(_default_handler)
    
    # 创建并设置格式化器
    fmt = NewLineFormatter(
        _FORMAT,  # 使用预定义的日志格式
        datefmt=_DATE_FORMAT  # 使用预定义的日期格式
    )
    _default_handler.setFormatter(fmt)
    
    # 禁用向父记录器传播,避免重复日志
    _root_logger.propagate = False
    
# 模块导入时初始化根日志器
# 由于Python的GIL(Global Interpreter Lock)保证模块只被导入一次
# 因此这个初始化操作是线程安全的
_setup_logger()

def init_logger(name: str) -> logging.Logger:
    """
    创建并初始化一个新的日志记录器。
    
    使用与根日志器相同的配置创建一个新的命名日志记录器。
    这个函数提供了一种创建具有一致配置的子日志器的方法。

    Args:
        name: 日志记录器的名称，通常使用模块的名称
              例如: "__main__", "myapp.module1"

    Returns:
        logging.Logger: 配置好的日志记录器实例

    Technical Details:
        1. 日志器配置:
           - DEBUG级别捕获所有日志
           - 使用全局默认处理器
           - 禁用向上传播
        
        2. 继承特性:
           - 与根日志器共享处理器
           - 保持格式一致性
           - 独立的日志级别

        3. 线程安全:
           - 支持并发访问
           - 共享处理器状态
           - 原子化操作

    Example:
        >>> # 创建模块级logger
        >>> logger = init_logger(__name__)
        >>> logger.info("Module initialized")
        INFO 12-23 14:30:45 module.py:10] Module initialized
        >>>
        >>> # 多模块使用示例
        >>> auth_logger = init_logger("auth")
        >>> db_logger = init_logger("database")
        >>> auth_logger.debug("Auth check")
        >>> db_logger.info("DB connected")

    Note:
        - 每个模块应该使用独立的logger
        - 保持命名层次结构
        - 避免logger名称冲突
        - 注意内存使用
    """
    # 创建新的日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别为DEBUG(捕获所有级别日志)
    logger.setLevel(logging.DEBUG)
    
    # 添加全局默认处理器
    # 这确保所有logger使用相同的输出格式和目标
    logger.addHandler(_default_handler)
    
    # 禁用向上传播
    # 防止日志消息被传递给父logger，避免重复日志
    logger.propagate = False
    
    return logger