"""
已知数组和目标值，在该数组中找到和为目标值的两个整数，返回数组下标
"""
# def TwoSum(nums, target):
#     lens = len(nums)
#     j = -1
#     for i in range(lens):
#         if (target - nums[i]) in nums:
#             if (nums.count(target - nums[i] == 0) == 1) & (target - nums[i] == nums[i]):
#                 continue
#             else:
#                 j = nums.index(target - nums[i], i + 1)
#                 break
#     if j > 0:
#         return [i, j]
#     else:
#         return []


# def TwoSum(nums, target):
#     lens = len(nums)
#     j = -1
#     for i in range(1, lens):
#         temp = nums[:i]#num1左半分片取数
#         if (target - nums[i]) in temp:
#             j = temp.index(target - nums[i])
#         if j >= 0:
#             return [j, i]




if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 17
    res = TwoSum(nums, target)
    print("result: \n", res)
