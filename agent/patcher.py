"""
Targeted code patcher.
Finds and replaces ONLY the relevant snippet in the target file,
creates backups, and computes unified diffs for human review.
"""

import difflib
import os
from pathlib import Path
from typing import Tuple, Optional


class CodePatcher:
    """Safely applies targeted search-and-replace patches to source files."""

    @staticmethod
    def read_file(file_path: str) -> Tuple[bool, str]:
        """Reads file content safely."""
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return True, f.read()
        except Exception as e:
            return False, f"Failed to read {file_path}: {e}"

    @staticmethod
    def write_file(file_path: str, content: str) -> Tuple[bool, str]:
        """Writes content to file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, "Success"
        except Exception as e:
            return False, f"Failed to write {file_path}: {e}"

    @classmethod
    def apply_patch(
        cls,
        file_path: str,
        search_block: str,
        replace_block: str,
    ) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Applies a targeted search-and-replace edit to the file.
        Returns: (success, message, backup_content, unified_diff)
        """
        ok, original_content = cls.read_file(file_path)
        if not ok:
            return False, original_content, None, None

        # Clean search and replace blocks from outer whitespace if needed
        clean_search = search_block.replace("\r\n", "\n")
        clean_replace = replace_block.replace("\r\n", "\n")
        clean_original = original_content.replace("\r\n", "\n")

        # 1. Direct exact match
        if clean_search in clean_original:
            occurrences = clean_original.count(clean_search)
            if occurrences > 1:
                # Still replace the first occurrence, but warn
                pass
            new_content = clean_original.replace(clean_search, clean_replace, 1)
        else:
            # 2. Tolerant matching: try stripping trailing whitespaces on each line
            new_content = cls._tolerant_replace(clean_original, clean_search, clean_replace)
            if new_content is None:
                return (
                    False,
                    f"Target search_block could not be found in '{file_path}'. "
                    f"Ensure the snippet matches the existing code lines exactly.",
                    original_content,
                    None,
                )

        # Write modified content
        ok_write, write_msg = cls.write_file(file_path, new_content)
        if not ok_write:
            return False, write_msg, original_content, None

        # Generate unified diff
        diff = cls.generate_diff(file_path, clean_original, new_content)
        return True, "Patch applied successfully", original_content, diff

    @staticmethod
    def _tolerant_replace(content: str, search: str, replace: str) -> Optional[str]:
        """
        Attempts line-based matching when subtle indentation or trailing space differences exist.
        """
        content_lines = content.splitlines()
        search_lines = [l.strip() for l in search.splitlines() if l.strip()]

        if not search_lines:
            return None

        # Find starting line match in content
        search_len = len(search_lines)
        for i in range(len(content_lines) - search_len + 1):
            window = [content_lines[i + j].strip() for j in range(search_len)]
            if window == search_lines:
                # Found matching line range [i : i + search_len]
                # Determine leading indentation of first matched line
                prefix_indent = ""
                original_first_line = content_lines[i]
                prefix_indent = original_first_line[: len(original_first_line) - len(original_first_line.lstrip())]

                # Adjust replace lines indentation if needed
                rep_lines = replace.splitlines()
                reconstructed = (
                    content_lines[:i]
                    + rep_lines
                    + content_lines[i + search_len :]
                )
                return "\n".join(reconstructed) + ("\n" if content.endswith("\n") else "")

        return None

    @staticmethod
    def generate_diff(file_path: str, old_content: str, new_content: str) -> str:
        """Generates unified diff string."""
        filename = os.path.basename(file_path)
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
        return "\n".join(diff)

    @classmethod
    def revert_file(cls, file_path: str, backup_content: str) -> Tuple[bool, str]:
        """Reverts a file back to its backup content."""
        return cls.write_file(file_path, backup_content)
