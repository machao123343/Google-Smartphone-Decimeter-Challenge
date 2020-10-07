"""
实现双向链表的操作
"""


class Node(object):
    """实现节点"""

    def __init__(self, item):
        self.item = item
        self.next = None


class SinCycLinkedlist(object):
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def length(self):
        if self.is_empty():
            return 0
        count = 1
        cur = self.head
        while cur.next != self.head:
            count = count + 1
            cur = cur.next
        return count

    def travel(self):
        if self.is_empty():
            return
        else:
            cur = self.head
            print(cur.item, end="\t")
            while cur.next != self.head:
                cur = cur.next
                print(cur.item, end="\t")

    def add(self, item):
        """
        头部添加节点
        """
        node = Node(item)
        if self.is_empty():
            self.head = node
            node.next = self.head  # 注意这种特殊的写法
        else:
            node.next = self.head
            cur = self.head  # 链表查找只能向前移动
            while cur.next != self.head:
                cur = cur.next
            cur.next = node  # cur移动指的是内存的移动，这里实际上已经插进去了，下一步是前移头节点标识
            self.head = node

    def append(self, item):
        node = Node(item)
        if self.is_empty():
            self.head = node
            node.next = self.head
        else:
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            cur.next = node
            node.next = self.head  # 在尾部加只是少了前移头节点标识这一步

    def insert(self, pos, item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            cur = self.head
            count = 0
            while count < (pos - 1):
                count = count + 1
                cur = cur.next
            node.next = cur.next
            cur.next = node

    def remove(self, item):
        if self.is_empty():
            return
        cur = self.head
        pre = None
        # 头节点为目标节点
        if cur.item == item:
            if cur.next != self.head:
                while cur.next != self.head:
                    cur = cur.next
                cur.next = self.next
                self.head = self.next
            else:
                self.head = None
        else:
            pre = self.head
            while cur.next != self.head:
                # 找到了要删除的节点
                if cur.item == item:
                    pre.next = cur.next
                    return
                else:
                    pre = cur
                    cur = cur.next
            # cur指向尾节点
            if cur.item == item:
                pre.next = cur.next
    def search(self, item):
        if self.is_empty():
            return False
        cur = self.head
        if cur.item == item:
            return True
        while cur.next != self.head:
            cur = cur.next
            if cur.item == item:
                return True
        return False

if __name__=='__main__':
    new = SinCycLinkedlist()
    new.add(1)
    new.add(2)
    new.append(3)
    new.insert(1, 4)
    new.travel()
