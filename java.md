# 7.整数反转(mid)
## 题目描述
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。
## 分析
xxxxxxx
## 代码
```java
class Solution {
    public int reverse(int x) {

    }
}
```
## 运行截图
![alt text](assets/image.png)

# 3.无重复字符的最长子串(mid)
## 题目描述
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串的长度。
## 分析
滑动窗口，当遇到重复字符时不断移动左侧left，直到没有重复字符。每轮记录是否当前窗口大于最长字串长度。
## 代码
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int len=s.length();
        int[] map=new int[130];
        int left=0;
        int ans=0;
        for(int i=0;i<len;i++){
            map[s.charAt(i)]++;
            while(map[s.charAt(i)]>1){
                map[s.charAt(left)]--;
                left++;
            }
            ans=Math.max(ans,i-left+1);
        }
        return ans;
    }
}
```
## 运行截图
![alt text](assets/3.png)

# 11.盛最多水的容器(mid)
## 题目描述
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。
## 分析
双指针，每次计算当前面积是否大于记录的，然后移动较小的边，贪心。
## 代码
```java
class Solution {
    public int maxArea(int[] height) {
        int n=height.length;
        int ans=0;
        int l=0,r=n-1;
        while(l<r){
            int cur=Math.min(height[l],height[r]);
            if(cur*(r-l)>ans){
                ans=cur*(r-l);
            }
            if(height[l]>height[r]){
                r--;
            }else{
                l++;
            }
        }
        return ans;
    }
}
```
## 运行截图
![alt text](assets/11.png)

# 23.合并 K 个升序链表(hard)
## 题目描述
给你一个链表数组，每个链表都已经按升序排列。
请你将所有链表合并到一个升序链表中，返回合并后的链表。
## 分析
首先利用类似归并的思想，从步长为1开始将相邻的链表两两连到一起，merge函数是将两个子链表连到一起。
## 代码
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int n=lists.length;
        if(n==0)return null;
        //类似归并
        for(int stride=1;stride<n;stride=stride*2){
            for(int i=0;i+stride<n;i+=stride*2){
                ListNode a=lists[i];
                ListNode b=lists[i+stride];
                lists[i]=merge(a,b);
            }
        }
        return lists[0];
    }
    public ListNode merge(ListNode head1,ListNode head2){
        ListNode head=new ListNode();
        ListNode memo=head;
        while(head1!=null&&head2!=null){
            if(head1.val<head2.val){
                head.next=head1;
                head1=head1.next;
            }else{
                head.next=head2;
                head2=head2.next;
            }
            head=head.next;
        }
        if(head1!=null){
            head.next=head1;
        }else{
            head.next=head2;
        }
        return memo.next;
    }
}
```
## 运行截图
![alt text](assets/23.png)

# 25.K 个一组翻转链表(hard)
## 题目描述
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
## 分析
模拟这一过程，创建多个临时链表节点来模拟。
## 代码
```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        //没有指针、->
        if(k==1)return head;
        ListNode vHead=new ListNode();
        vHead.next=null;
        ListNode h=vHead;
        while(head != null){
            ListNode p=head;
            ListNode first=p;
            ListNode q;
            int i=0;
            while(head != null && i<k){
                head=head.next;
                i++;
                //head移动
            }
            if(i<k){
                h.next=first;
                break;
            }
            for(int j=0;j<k;j++){
                q=p.next;
                p.next=h.next;
                h.next=p;
                p=q;
            }
            h=first;
        }
        return vHead.next;
    }
}
```
## 运行截图
![alt text](assets/25.png)

# 42.接雨水(hard)
## 题目描述
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
## 分析
从左至右遍历一遍，存储遇到的最高值；再从右往左一遍。最后计算两边最高值的较小值与当前格子的值只差计入ans。
## 代码
```java
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int[] a=new int[n];
        for(int i=0;i<n;i++){
            a[i]=height[i];
        }
        int cur_high=0;
        for(int i=0;i<n;i++){//from left
            cur_high=Math.max(cur_high,height[i]);
            a[i]=cur_high;
        }
        cur_high=0;
        for(int i=n-1;i>=0;i--){//from right
            cur_high=Math.max(cur_high,height[i]);
            a[i]=Math.min(a[i],cur_high);
        }
        int ans=0;
        for(int i=0;i<n;i++){
            ans+=a[i]-height[i];
        }
        return ans;
    }
}
```
## 运行截图
![alt text](assets/42.png)

# 239.滑动窗口最大值(hard)
## 题目描述
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。返回 滑动窗口中的最大值 。
## 分析
使用双端队列，存储一个单调递减的数，遍历数组，如果大于队尾则遍历弹出队尾。每次队尾的值就是所求。
## 代码
```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n=nums.length;
        Deque<Integer> deque=new LinkedList<Integer>();
        for(int i=0;i<k;i++){
            while(!deque.isEmpty() && nums[i]>=nums[deque.peekLast()]){
                deque.pollLast();
            }
            deque.offerLast(i);
        }
        int[] ans=new int[n-k+1];
        ans[0]=nums[deque.peekFirst()];
        for(int i=k;i<n;i++){
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            if(deque.peekFirst()<=i-k){
                deque.pollFirst();//滑出窗口左侧
            }
            ans[i-k+1]=nums[deque.peekFirst()];
        }
        return ans;
    }
}
```
## 运行截图
![alt text](assets/239.png)

# 283.移动零(easy)
## 题目描述
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意 ，必须在不复制数组的情况下原地对数组进行操作。
## 分析
遍历一遍即可，将遇到的非零数字依次填写，最后把index后面的位置补0
## 代码
```java
class Solution {
    public void moveZeroes(int[] nums) {
        int n=nums.length;
        int index=0;
        for(int i=0;i<n;i++){
            if(nums[i]!=0){//把非0的移到顶
                int temp=nums[i];
                nums[i]=nums[index];
                nums[index]=temp;
                index++;
            }
        }
        for(;index<n;index++){
            nums[index]=0;
        }
    }
}
```
## 运行截图
![alt text](assets/283.png)

# 560.和为 K 的子数组(mid)
## 题目描述
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
## 分析
利用哈希表存储遇到的前缀和，如果当前前缀和减去k的结果在map中出现过，则这之间的数字之和为k，加入答案中。不断更新前缀和并加入哈希表。
## 代码
```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n=nums.length;
        int ans=0;
        int pre=0;
        HashMap<Integer,Integer> mp=new HashMap<>();
        mp.put(0,1);
        for(int i=0;i<n;i++){
            pre+=nums[i];
            if(mp.containsKey(pre-k)){
                ans+=mp.get(pre-k);
            }
            mp.put(pre,mp.getOrDefault(pre,0)+1);//getOrDefault
        }
        return ans;
    }
}
```
## 运行截图
![alt text](assets/560.png)
