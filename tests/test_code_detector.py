"""
Tests for code detection and language identification.
"""

from __future__ import annotations

import unittest

from vidtrace.vision.code_detector import (
    CodeDetector,
    ProgrammingLanguage,
    detect_language,
    is_code_block,
)


class TestDetectLanguage(unittest.TestCase):
    """Tests for detect_language()."""

    def test_python(self) -> None:
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def __init__(self):
        self.result = 0
"""
        lang, score = detect_language(code)
        assert lang == ProgrammingLanguage.PYTHON.value
        assert score > 4.0

    def test_javascript(self) -> None:
        code = """
const express = require('express');
const app = express();

function handleRequest(req, res) {
    console.log('Request received');
    res.send('Hello');
}
"""
        lang, score = detect_language(code)
        assert lang == ProgrammingLanguage.JAVASCRIPT.value

    def test_java(self) -> None:
        code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
"""
        lang, score = detect_language(code)
        assert lang == ProgrammingLanguage.JAVA.value

    def test_cpp(self) -> None:
        code = """
#include <iostream>
using namespace std;

int main() {
    cout << "Hello" << endl;
    return 0;
}
"""
        lang, score = detect_language(code)
        assert lang == ProgrammingLanguage.CPP.value

    def test_sql(self) -> None:
        code = """
SELECT users.name, orders.total
FROM users
JOIN orders ON users.id = orders.user_id
WHERE orders.total > 100
GROUP BY users.name
"""
        lang, score = detect_language(code)
        assert lang == ProgrammingLanguage.SQL.value

    def test_shell(self) -> None:
        code = """
#!/bin/bash
echo "Installing dependencies"
pip install -r requirements.txt
git clone https://github.com/example/repo.git
"""
        lang, score = detect_language(code)
        assert lang == ProgrammingLanguage.SHELL.value

    def test_empty(self) -> None:
        lang, score = detect_language("")
        assert lang == ProgrammingLanguage.UNKNOWN.value
        assert score == 0.0

    def test_plain_text(self) -> None:
        text = "This is a regular English sentence about the weather today."
        lang, score = detect_language(text)
        assert score < 4.0  # Below threshold


class TestIsCodeBlock(unittest.TestCase):
    """Tests for is_code_block()."""

    def test_python_code(self) -> None:
        code = "def main():\n    x = 42\n    return x"
        assert is_code_block(code) is True

    def test_plain_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        assert is_code_block(text) is False

    def test_empty(self) -> None:
        assert is_code_block("") is False

    def test_mixed_content(self) -> None:
        code = "import os\npath = os.getcwd()\nprint(path)"
        assert is_code_block(code) is True

    def test_single_line_code(self) -> None:
        code = "const x = 42;"
        assert is_code_block(code) is True


class TestCodeDetector(unittest.TestCase):
    """Tests for CodeDetector."""

    def setUp(self) -> None:
        self.detector = CodeDetector()

    def test_detect_python_region(self) -> None:
        lines = [
            "def fibonacci(n):",
            "    if n <= 1:",
            "        return n",
            "    return fibonacci(n-1) + fibonacci(n-2)",
        ]
        regions = self.detector.detect_regions(lines)
        assert len(regions) >= 1
        assert regions[0].language == ProgrammingLanguage.PYTHON.value

    def test_no_code_in_text(self) -> None:
        lines = [
            "Welcome to the lecture",
            "Today we will discuss algorithms",
            "Please open your textbooks",
        ]
        regions = self.detector.detect_regions(lines)
        assert len(regions) == 0

    def test_empty_input(self) -> None:
        regions = self.detector.detect_regions([])
        assert regions == []

    def test_mixed_code_and_text(self) -> None:
        lines = [
            "This is the title slide",
            "Introduction to Python",
            "def hello():",
            "    print('Hello')",
            "    return True",
            "Thank you for watching",
        ]
        regions = self.detector.detect_regions(lines)
        # Should find the code block in the middle.
        assert len(regions) >= 1

    def test_indentation_detected(self) -> None:
        lines = [
            "class MyClass:",
            "    def __init__(self):",
            "        self.value = 0",
        ]
        regions = self.detector.detect_regions(lines)
        if regions:
            assert regions[0].indentation_preserved is True


if __name__ == "__main__":
    unittest.main()
