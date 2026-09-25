from codeforces import extract_code


def test_extract_code_takes_the_python_block():
    text = "Here is my idea.\n```python\nprint(input())\n```\nDone."
    assert extract_code(text) == "print(input())\n"


def test_extract_code_takes_the_last_block_when_there_are_several():
    text = "```python\nprint(1)\n```\nFixed:\n```python\nprint(2)\n```"
    assert extract_code(text) == "print(2)\n"


def test_extract_code_accepts_an_unlabelled_fence():
    assert extract_code("```\nprint(3)\n```") == "print(3)\n"


def test_extract_code_is_none_without_a_fence():
    assert extract_code("print(4)") is None


def test_extract_code_for_cpp_takes_the_cpp_block():
    text = "```python\nprint(1)\n```\n```cpp\nint main() {}\n```"
    assert extract_code(text, "cpp") == "int main() {}\n"
    assert extract_code(text, "python") == "print(1)\n"
