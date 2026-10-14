from urllib.request import urlopen
with urlopen('https://docs.python.org/3/') as response:
    for line in response:
        line = line.decode()             # Convert bytes to a str
        if 'updated' in line:
            print(line.rstrip())         # Remove trailing newline



import smtplib
server = smtplib.SMTP('localhost')
server.sendmail('soothsayer@example.org', 'jcaesar@example.org',
"""To: jcaesar@example.org
From: soothsayer@example.org

Beware the Ides of March.
""")
server.quit()


# dates are easily constructed and formatted
import datetime as dt
now = dt.date.today()
print(now)
print(now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))
# dates support calendar arithmetic
birthday = dt.date(2004, 1, 4)
age = now - birthday
print(age.days)

import zlib
s = b'witch which has which witches wrist watch'
print(len(s))
t = zlib.compress(s)
print(len(t))
zlib.decompress(t)
print(zlib.crc32(s))

from timeit import Timer
Timer('t=a; a=b; b=t', 'a=1; b=2').timeit()

Timer('a,b = b,a', 'a=1; b=2').timeit()


def average(values):
    """计算数字列表的算术平均值

    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)

import doctest
doctest.testmod()   # 自动验证嵌入式测试

import unittest

class TestStatisticalFunctions(unittest.TestCase):

    def test_average(self):
        self.assertEqual(average([20, 30, 70]), 40.0)
        self.assertEqual(round(average([1, 5, 7]), 1), 4.3)
        with self.assertRaises(ZeroDivisionError):
            average([])
        with self.assertRaises(TypeError):
            average(20, 30, 70)

unittest.main()  # 从命令行调用时会执行所有测试

