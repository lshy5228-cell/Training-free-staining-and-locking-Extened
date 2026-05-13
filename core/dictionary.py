# -*- coding: utf-8 -*-

class BuiltInDictionary:
    """
    Core fallback dictionary containing essential UI mappings.
    In a production GitHub repo, this list would extend to 500+ entries.
    """
    TRANSLATIONS = {
        # Global Navigation
        "Settings": "设置",
        "Scans": "扫描",
        "Policies": "策略",
        "Plugin Rules": "插件规则",
        "Territories": "扫描区域",
        "Logout": "退出登录",
        
        # Scan Status
        "Completed": "已完成",
        "Running": "运行中",
        "Pending": "等待中",
        "Canceled": "已取消",
        "Critical": "紧急",
        "High": "高危",
        "Medium": "中危",
        "Low": "低危",
        "Info": "信息",
        
        # Details
        "Vulnerabilities": "漏洞",
        "Description": "描述",
        "Solution": "解决方案",
        "See Also": "参考链接",
        "Output": "扫描输出",
        "Hosts": "主机",
        "Asset": "资产",
        "History": "历史记录",
        
        # Buttons/Actions
        "New Scan": "新建扫描",
        "Launch": "启动",
        "Export": "导出",
        "Save": "保存",
        "Cancel": "取消",
        "Trash": "回收站"
    }

    @classmethod
    def get_count(cls):
        return len(cls.TRANSLATIONS)
