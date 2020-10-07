import time

# start_time = time.time()
# for a in range(0, 1001):
#     for b in range(0, 1001):
#         for c in range(0, 1001):
#             if a**2 + b**2 == c**2 and a + b + c == 1000:
#                 print("a, b, c: %d, %d, %d"%(a, b, c))
#
# end_time = time.time()
# print("time is : %f"%(end_time - start_time))


# start_time = time.time()
# for a in range(0, 1001):
#     for b in range(0, 1001 - a):
#         c = 1000 - a - b
#         if a**2 + b**2 == c**2:
#             print("a, b, c: %d, %d, %d" % (a, b, c))
# end_time = time.time()
# print("time is :%f"%(end_time - start_time))

"""
时间复杂度差了一个n的级别
"""

"""
顺序表--直接访问通过计算的物理地址，时间复杂度为1，下面为实现
存储结构：一体结构，分离结构（数组）
"""

"""
python链表实现
"""


# 实现节点
class SingleNode(object):
    def __init__(self, item):
        self.item = item
        self.next = None


class SingleLinkList(object):
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def length(self):
        cur = self.head
        count = 0
        while cur is not None:
            count = count + 1
            cur = cur.next
        return count

    def travel(self):
        cur = self.head
        while cur is not None:
            print(cur.item, end="\t")
            cur = cur.next

    # 头部添加节点
    def add(self, item):
        """头部添加元素"""
        node = SingleNode(item)
        node.next = self.head
        self.head = node

    def append(self, item):
        node = SingleNode(item)
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = SingleNode(item)
            count = 0
            # pre用来指向指定位置pos的前一个位置pos-1，初始从头节点开始移动到指定位置
            pre = self.head
            while count < (pos - 1):
                count = count + 1
                pre = pre.next
            node.next = pre.next
            pre.next = node

    def search(self, item):
        cur = self.head
        while cur is not None:
            if cur.item == item:
                return True
            cur = cur.next
        return False


if __name__ == "__main__":
    ll = SingleLinkList()
    ll.add(1)
    ll.add(2)
    ll.append(3)
    ll.insert(2, 4)
    print("length:", ll.length())
    ll.travel()
    print(ll.search(3))
    print(ll.search(5))
