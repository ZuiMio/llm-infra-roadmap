"""
Day 01 - Python基礎の追加学習

このファイルでは、Pythonの基本的な構文について追加で学習する。

学習内容:
- 基本的な算術演算
- if / elif / else
- for / while
- range()
- break / continue
- for-else
- 辞書の操作
- pass
- match / case
- クラスパターン
- 関数とデフォルト引数
- whileによる入力処理
"""


# ============================================================
# 1. 基本的な算術演算
# ============================================================

def basic_math_operations():
    """基本的な算術演算とraw文字列を確認する。"""

    print("15 / 3 =", 15 / 3)       # 通常の除算
    print("15 // 3 =", 15 // 3)     # 整数除算
    print("15 % 3 =", 15 % 3)       # 余り
    print("15 ** 2 =", 15 ** 2)     # べき乗

    # r を文字列の前に付けると、\n や \t などの
    # エスケープシーケンスを特殊文字として解釈しない。
    print(r"C:\this\name")


# ============================================================
# 2. if / elif / else
# ============================================================

def if_statement_example():
    """条件分岐の基本的な使い方を確認する。"""

    print("1 + 1 == 2:", 1 + 1 == 2)

    if True:
        print("It's true!")
    else:
        print("It's false!")

    x = -1

    if x < 0:
        x = 0
        print("Negative changed to zero")
    elif x == 0:
        print("Zero")
    elif x == 1:
        print("Single")
    else:
        print("More")


# ============================================================
# 3. 辞書を反復しながらデータを処理する
# ============================================================

users = {
    "Hans": "active",
    "Éléonore": "inactive",
    "景太郎": "active",
}


def remove_inactive_users(users):
    """
    非アクティブなユーザーを元の辞書から削除する。

    反復中に元の辞書を直接変更すると問題が発生するため、
    users.copy() を使ってコピーを反復する。
    """

    for user, status in users.copy().items():
        if status == "inactive":
            del users[user]


def get_active_users(users):
    """
    active のユーザーだけを含む新しい辞書を作成して返す。

    元の辞書は変更しない。
    """

    active_users = {}

    for user, status in users.items():
        if status == "active":
            active_users[user] = status

    return active_users


# ============================================================
# 4. range()
# ============================================================

def range_example():
    """range() の基本的な使い方を確認する。"""

    print("range(5):")

    for i in range(5):
        print(i)

    print("range(5, 10):", list(range(5, 10)))
    print("range(0, 10, 3):", list(range(0, 10, 3)))
    print("range(-10, -100, -30):", list(range(-10, -100, -30)))

    # 0 + 1 + 2 + 3
    print("sum(range(4)):", sum(range(4)))


# ============================================================
# 5. break と continue
# ============================================================

def break_continue_example():
    """break と continue の動作を確認する。"""

    print("break の例:")

    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                print(f"{n} は {x} * {n // x} と等しい")
                break

    print("\ncontinue の例:")

    for num in range(2, 10):
        if num % 2 == 0:
            print(f"偶数 {num} を見つけた")
            continue

        print(f"奇数 {num} を見つけた")


# ============================================================
# 6. for-else
# ============================================================

def for_else_example():
    """
    for 文に付けられる else の使い方を確認する。

    break が実行されずに for ループが最後まで終了した場合だけ、
    else ブロックが実行される。
    """

    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                print(f"{n} は {x} * {n // x} と等しい")
                break
        else:
            # 約数が見つからず、breakされなかった場合
            print(f"{n} は素数です")


# ============================================================
# 7. pass
# ============================================================

class MyEmptyClass:
    """
    まだ処理を実装していない空のクラス。

    pass は「何もしない」文であり、
    構文上何かを書く必要がある場所で使用できる。
    """

    pass


# ============================================================
# 8. match / case
# ============================================================

def http_error(status):
    """HTTPステータスコードに応じたメッセージを返す。"""

    match status:
        case 400:
            return "Bad request"

        case 404:
            return "Not found"

        case 418:
            return "I'm a teapot"

        # | を使うと複数の値にマッチできる。
        case 401 | 403:
            return "Not allowed"

        # _ はどのパターンにもマッチするワイルドカード。
        case _:
            return "Something's wrong with the internet"


# ============================================================
# 9. クラスパターンによるmatch
# ============================================================

class Point:
    """2次元座標を表すクラス。"""

    # match 文で Point(x, y) のように
    # 位置パターンを使えるようにする。
    __match_args__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def where_is(point):
    """Pointオブジェクトがどこにあるかを判定する。"""

    match point:
        case Point(x=0, y=0):
            print("原点")

        case Point(x=0, y=y):
            print(f"Y軸上: Y={y}")

        case Point(x=x, y=0):
            print(f"X軸上: X={x}")

        case Point():
            print("それ以外の座標")

        case _:
            print("Pointではありません")


def where_are_points(points):
    """Pointオブジェクトのリストをパターンマッチングする。"""

    match points:
        case []:
            print("点がありません")

        case [Point(0, 0)]:
            print("原点だけがあります")

        case [Point(x, y)]:
            print(f"1つの点があります: ({x}, {y})")

        case [Point(0, y1), Point(0, y2)]:
            print(f"Y軸上に2つの点があります: Y={y1}, Y={y2}")

        case _:
            print("その他のパターンです")


# ============================================================
# 10. Fibonacci数列
# ============================================================

def fib(n):
    """
    n 未満のフィボナッチ数を出力する。

    Fibonacci数列:
    0, 1, 1, 2, 3, 5, 8, ...
    """

    a, b = 0, 1

    while a < n:
        print(a, end=" ")

        # Pythonでは複数の変数を同時に更新できる。
        a, b = b, a + b

    print()


# ============================================================
# 11. デフォルト引数とユーザー入力
# ============================================================

def ask_ok(prompt, retries=4, reminder="再試行してください!"):
    """
    ユーザーから yes / no の入力を受け取る。

    有効な入力:
        yes: y, ye, yes
        no : n, no, nop, nope

    無効な入力が続き、retriesを使い切った場合は
    ValueErrorを発生させる。
    """

    while True:
        reply = input(prompt)

        if reply in {"y", "ye", "yes"}:
            return True

        if reply in {"n", "no", "nop", "nope"}:
            return False

        retries -= 1

        if retries < 0:
            raise ValueError("無効なユーザー入力です")

        print(reminder)


# ============================================================
# 12. ミュータブルなデフォルト引数を避ける
# ============================================================

def append_to_list(a, L=None):
    """
    値 a をリスト L に追加して返す。

    デフォルト引数として [] を直接使用すると、
    同じリストが複数回の関数呼び出しで共有される。

    そのため、デフォルト値には None を使用し、
    関数内部で新しいリストを作成する。
    """

    if L is None:
        L = []

    L.append(a)

    return L


# ============================================================
# main
# ============================================================

def main():
    """このファイルの各サンプルを実行する。"""

    print("=== 1. 基本的な算術演算 ===")
    basic_math_operations()

    print("\n=== 2. if文 ===")
    if_statement_example()

    print("\n=== 3. 辞書の操作 ===")

    sample_users = users.copy()

    print("変更前:", sample_users)
    remove_inactive_users(sample_users)
    print("inactive削除後:", sample_users)

    print("activeユーザー:", get_active_users(users))

    print("\n=== 4. range ===")
    range_example()

    print("\n=== 5. break / continue ===")
    break_continue_example()

    print("\n=== 6. for-else ===")
    for_else_example()

    print("\n=== 7. match / case ===")

    for status in [400, 401, 403, 404, 418, 500]:
        print(status, "->", http_error(status))

    print("\n=== 8. Pointのパターンマッチング ===")

    where_is(Point(0, 0))
    where_is(Point(0, 5))
    where_is(Point(3, 0))
    where_is(Point(3, 5))
    where_is("not a point")

    where_are_points([])
    where_are_points([Point(0, 0)])
    where_are_points([Point(3, 4)])
    where_are_points([Point(0, 3), Point(0, 8)])

    print("\n=== 9. Fibonacci数列 ===")
    fib(2000)

    print("\n=== 10. デフォルト引数 ===")

    print(append_to_list(1))
    print(append_to_list(2))
    print(append_to_list(3, [10, 20]))


if __name__ == "__main__":
    main()