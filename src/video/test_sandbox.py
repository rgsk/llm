import pytest

from sandbox import NO_NET, judge, run, same

ADD = "a, b = map(int, input().split())\nprint(a + b)\n"


def test_run_feeds_stdin_and_captures_stdout():
    r = run(ADD, "2 3\n")
    assert (r.status, r.stdout) == ("ok", "5\n")


def test_crash_is_an_error_with_the_traceback():
    r = run("print(1 // 0)", "")
    assert r.status == "error"
    assert "ZeroDivisionError" in r.stderr


def test_infinite_loop_is_killed():
    assert run("while True: pass", "", timeout=0.5).status == "timeout"


def test_memory_cap_turns_a_huge_list_into_an_error():
    r = run("x = [0] * 10**9", "", mem_mb=256)
    assert r.status == "error"
    assert "MemoryError" in r.stderr


@pytest.mark.skipif(not NO_NET, reason="no user namespaces on this kernel")
def test_no_network():
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)"
    r = run(code, "", timeout=5)
    assert r.status == "error"
    assert "unreachable" in r.stderr


def test_same_ignores_whitespace_but_not_tokens():
    assert same("5\n", "5")
    assert same("1 2\n3", "1\n2 3\n")
    assert not same("YES", "yes")
    assert not same("1 2", "1 2 3")


def test_judge_accepts_when_every_test_matches():
    v = judge(ADD, [("2 3\n", "5\n"), ("10 -4\n", "6")])
    assert (v.passed, v.status, v.tests_run) == (True, "accepted", 2)


def test_judge_stops_at_the_first_wrong_answer():
    sub = "a, b = map(int, input().split())\nprint(a - b)\n"
    v = judge(sub, [("5 0\n", "5"), ("2 3\n", "5"), ("1 1\n", "2")])
    assert (v.passed, v.status, v.tests_run) == (False, "wrong", 2)


ADD_CPP = "#include <iostream>\nint main() { long long a, b; std::cin >> a >> b; std::cout << a + b; }\n"


def test_cpp_compiles_once_and_runs_every_test():
    v = judge(ADD_CPP, [("2 3\n", "5\n"), ("10 -4\n", "6")], lang="cpp")
    assert (v.passed, v.status, v.tests_run) == (True, "accepted", 2)


def test_cpp_that_does_not_compile_runs_no_tests():
    v = judge("int main() { return x; }", [("", "")], lang="cpp")
    assert (v.passed, v.status, v.tests_run) == (False, "compile_error", 0)


def test_cpp_compiler_error_is_in_stderr():
    r = run("int main() { return x; }", "", lang="cpp")
    assert r.status == "compile_error"
    assert "was not declared in this scope" in r.stderr


def test_output_that_is_not_utf8_is_a_wrong_answer_not_a_crash():
    code = "#include <cstdio>\nint main() { putchar(0x9c); }\n"
    v = judge(code, [("", "1")], lang="cpp")
    assert (v.passed, v.status) == (False, "wrong")
