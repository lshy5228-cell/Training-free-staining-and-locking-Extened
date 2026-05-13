import re
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

class PatchEngine:
    """
    Advanced Regex Engine for Context-Aware Matching in obfuscated JS.
    """
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.backup_ext = ".original.bak"

    def apply_patch(self, file_path):
        """
        Applies translations to a specific JS file using a non-destructive approach.
        """
        if not os.path.exists(file_path):
            return 0

        # Backup Logic
        backup_path = file_path + self.backup_ext
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            logging.debug(f"Created backup for {os.path.basename(file_path)}")

        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                content = f.read()

            hit_count = 0
            # Tiered replacement: matching strings within quotes only
            for eng, chn in self.dictionary.items():
                pattern = rf'([\"\']){re.escape(eng)}([\"\'])'
                new_content, count = re.subn(pattern, rf'\1{chn}\2', content)
                if count > 0:
                    content = new_content
                    hit_count += count

            if hit_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return hit_count
        except Exception as e:
            logging.error(f"Failed to patch {file_path}: {e}")
        
        return 0
