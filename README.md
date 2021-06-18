





asaa

#### 1、leetcode 144题：给你二叉树的根节点 root ，返回它节点值的 前序 遍历

输入：root = [1,null,2,3]
输出：[1,2,3]
code:

```
class Solution {
    List<Integer>  ans = new  ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root) {
        if(root == null){
            return ans;
        }
        ans.add(root.val);
        preorderTraversal(root.left);
        preorderTraversal(root.right);
        return ans
    }
}


class Solution {
    List<Integer>ans = new ArrayList<Integer>();
    public List<Integer> preorderTraversal(TreeNode root) {
        if(root == null)
            return ans;
        Stack<TreeNode> stack = new Stack<TreeNode> ();
        stack.push(root);
        while(!stack.isEmpty()){
            TreeNode node = stack.pop();
            ans.add(node.val);
            if(node.right!= null) 
                stack.push(node.right);
            if(node.left!= null) 
                stack.push(node.left);
        }
        return ans;
    }
}


```

#### 2、leetcode 94题 二叉树的中序遍历

输入：root = [1,null,2,3]
输出：[1,2,3]

```
递归版本
class Solution {
    List<Integer> ans = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        if(root == null)
            return ans;
        inorderTraversal(root.left);
        ans.add(root.val);
        inorderTraversal(root.right);
        return ans;
    }
}
```

```
  class Solution {
    List<Integer>  ans = new  ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> stack = new Stack<>();
            while (!stack.isEmpty() || root != null) {
                if (root != null) {
                    stack.push(root);
                    root = root.left;
                } else {
                    root = stack.pop();
                    ans.add(root.val);
                    root = root.right;
                }
            }
        }
        return ans;
    }
}
```



#### 3、leetcode 94题 二叉树的后序遍历

```
迭代
class Solution {
    List<Integer> ans = new ArrayList<>();
    public List<Integer> postorderTraversal(TreeNode root) {
        if(root == null)
            return ans;
        postorderTraversal(root.left);
        postorderTraversal(root.right);
        ans.add(root.val);
        return ans;
    }
}
```



```
非递归版本
class Solution {
    List<Integer> ans = new ArrayList<>();
    Stack<Integer> mystack = new Stack<>();
    public List<Integer> postorderTraversal(TreeNode root) {
        if(root == null)
            return ans;
        Stack<TreeNode> stack =  new Stack<>();
        stack.push(root);
        while(!stack.empty()){
            root =stack.pop();
            mystack.push(root.val);
            if(root.left != null) {
                stack.push(root.left);
            }
            if(root.right != null) {
                stack.push(root.right);
            }
        }
        while(!mystack.empty()){
            ans.add(mystack.pop());
        }
        return ans;
    }
}
```

#### 4、leetcode 102题 二叉树的层序遍历

```
递归版本
class Solution {
    public void addLevel(List<List<Integer>> list, int level, TreeNode root) {
        if(root == null)
           return;
        if(list.size()-1 < level)
           list.add(new ArrayList<>());
        list.get(level).add(root.val );
        addLevel(list, level+1 , root.left);
        addLevel(list, level+1 , root.right);
    }
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans  =new ArrayList<>();
        addLevel(ans, 0 ,root);
        return ans;
    }
}
```



```
非递归
public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans  =new ArrayList<>();
        if(root == null)   return ans;
        Queue<TreeNode> que = new LinkedList<>();
        que.add(root);
        int size = 0;
        int level =0;
        TreeNode node = null;
        while(!que.isEmpty()){
            size = que.size();
            ans.add(new ArrayList<>());
            for(int i=0;i<size;i++){
                node = que.poll();
                ans.get(level).add(node.val);
                if(node.left!= null) que.add(node.left);
                if(node.right!= null) que.add(node.right);
            }
            level++;
        }
        return ans;
}
```



#### 5、排序算法

##### 5.1冒泡排序：复杂度 O（N^2）稳定

```
 public static  void myBubbleSort(int []arr){
        for(int i= arr.length -1;i>0 ;i--){
            for (int j = 0; j < i; j++) {
                if(arr[j] > arr[j+1])
                    swap(arr,j,j+1);
            }
        }
    }
```



5.2选择排序：复杂度 O（N^2）不稳定；数据  5 8 5 2 9 



```
public static void selectionSort(int[] arr) {
		if (arr == null || arr.length < 2) {
			return;
		}
		for (int i = 0; i < arr.length - 1; i++) {
			int minIndex = i;
			for (int j = i + 1; j < arr.length; j++) {
				minIndex = arr[j] < arr[minIndex] ? j : minIndex;
			}
			swap(arr, i, minIndex);
		}
}


```



##### 5.3 堆排序：复杂度 O（NlogN）不稳定；



```
public static void heapSort(int[] arr) {
		if (arr == null || arr.length < 2) {
			return;
		}
		for (int i = 0; i < arr.length; i++) {
			heapInsert(arr, i);
		}
		int size = arr.length;
		swap(arr, 0, --size);
		while (size > 0) {
			heapify(arr, 0, size);
			swap(arr, 0, --size);
		}
	}

public static void heapInsert(int[] arr, int index) {
		while (arr[index] > arr[(index - 1) / 2]) {
			swap(arr, index, (index - 1) / 2);
			index = (index - 1) / 2;
		}
}
	
public static void heapify(int[] arr, int index, int size) {
		int left = index * 2 + 1;
		while (left < size) {
			int largest = left + 1 < size && arr[left + 1] > arr[left] ? left + 1 : left;
			largest = arr[largest] > arr[index] ? largest : index;
			if (largest == index) {
				break;
			}
			swap(arr, largest, index);
			index = largest;
			left = index * 2 + 1;
		}
}


```







##### 5.4快速排序：复杂度 O（NlogN）不稳定（可以稳定，01stable sort ）；

```
public static void quickSort(int[] arr) {
		if (arr == null || arr.length < 2) {
			return;
		}
		quickSort(arr, 0, arr.length - 1);
	}

	public static void quickSort(int[] arr, int l, int r) {
		if (l < r) {
			swap(arr, l + (int) (Math.random() * (r - l + 1)), r);
			int[] p = partition(arr, l, r);
			quickSort(arr, l, p[0] - 1);
			quickSort(arr, p[1] + 1, r);
		}
	}
	
	public static int[] partition(int[] arr, int l, int r) {
		int less = l - 1;
		int more = r;
		while (l < more) {
			if (arr[l] < arr[r]) {
				swap(arr, ++less, l++);
			} else if (arr[l] > arr[r]) {
				swap(arr, --more, l);
			} else {
				l++;
			}
		}
		swap(arr, more, r);
		return new int[] { less + 1, more };
	}


```



##### 5.4.1 LeetCode 75 颜色分类问题。类似于荷兰国旗问题

```
class Solution {
  public void swap(int[] nums,int i,int j){
    int temp = nums[i];
    nums[i]= nums[j];
    nums[j] = temp;
  }
  public void sortColors(int[] nums) {
     int less = -1, more = nums.length;
     int i= 0;
     while(i < more){
       if(nums[i]==0) {
         swap(nums, ++less, i++);
       }else if(nums[i] == 2){
         swap(nums, --more,i);
       }else if(nums[i] == 1){
         i++;
       }
     }
  }
}
```



##### 5.5归并排序：复杂度 O（NlogN）稳定

```
public static void mergeSort(int[] arr) {
		if (arr == null || arr.length < 2) {
			return;
		}
		mergeSort(arr, 0, arr.length - 1);
	}

	public static void mergeSort(int[] arr, int l, int r) {
		if (l == r) {
			return;
		}
		int mid = l + ((r - l) >> 1);
		mergeSort(arr, l, mid);
		mergeSort(arr, mid + 1, r);
		merge(arr, l, mid, r);
	}
	
	public static void merge(int[] arr, int l, int m, int r) {
		int[] help = new int[r - l + 1];
		int i = 0;
		int p1 = l;
		int p2 = m + 1;
		while (p1 <= m && p2 <= r) {
			help[i++] = arr[p1] < arr[p2] ? arr[p1++] : arr[p2++];
		}
		while (p1 <= m) {
			help[i++] = arr[p1++];
		}
		while (p2 <= r) {
			help[i++] = arr[p2++];
		}
		for (i = 0; i < help.length; i++) {
			arr[l + i] = help[i];
		}
	}


```



```
非递归
class Solution {
    List<Integer>  ans = new  ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root) {
        if(root == null){
            return ans;
        }
        Stack<TreeNode> stack =  new Stack<>();
        stack.push(root);
        //TreeNode node;
        while(!stack.empty()){
            root =stack.pop();
            ans.add(root.val);
            if(root.right != null) {
                stack.push(root.right);
            }
            if(root.left != null) {
                stack.push(root.left);
            }
        }
        return ans;
    }
}
```



#### 6、leetcode 剑指Offer 41数据流的中位数

```

class MedianFinder {

    /** initialize your data structure here. */
    PriorityQueue<Integer> maxHeap;
    PriorityQueue<Integer> minHeap;
    public MedianFinder() {
         maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
    
        minHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1.compareTo(o2);
            }
        });
    }
    
    public void addNum(int num) {
        if(maxHeap.isEmpty())   {
            maxHeap.add(num);
            return;
        }
        if(num > maxHeap.peek())
            minHeap.add(num);
        else
            maxHeap.add(num);
        while(   maxHeap.size() - minHeap.size()  >1){
            minHeap.add(maxHeap.poll());
        }
        while(  minHeap.size() - maxHeap.size()  >= 1){
            maxHeap.add(minHeap.poll());
        }
    }
    
    public double findMedian() {
        double ans =0;
        int size = maxHeap.size() + minHeap.size();
        ans = (size%2 ==0)? (maxHeap.peek() + minHeap.peek())/2.0f: maxHeap.peek();
        return ans;
    }
}

```

#### 12月10号

#### 7、桶排序

假设数据只在（0-200之间，数据波动比较小）比如年龄等。开辟一个大小为200的数组，去进行计数。最后将数组的每个元素的值恢复成排序。





	public static void bucketSort(int[] arr) {
		if (arr == null || arr.length < 2) {
			return;
		}
		int max = Integer.MIN_VALUE;
		//找到最大的值
		for (int i = 0; i < arr.length; i++) {
			max = Math.max(max, arr[i]);
		}
		int[] bucket = new int[max + 1];
		for (int i = 0; i < arr.length; i++) {
			bucket[arr[i]]++;
		}
		int i = 0;
		//将计数后的数组恢复回去
		for (int j = 0; j < bucket.length; j++) {
			while (bucket[j]-- > 0) {
				arr[i++] = j;
			}
		}
	}
##### 7.1 桶排序相关的题 leetcode 164 最大间距 有bug

思路：有n个数，遍历获得最大最小值，申请n+1个桶，将每个数放进桶中。维护每个桶的最大最小值遍历一次。如果是访问第一个

获得答案

```
有点Bug需要调试一下
public static int maximumGap(int[] nums) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for(int i=0;i<nums.length;i++){
            max = nums[i]>max ? nums[i]:max;
            min = nums[i]<min ? nums[i]:min;
        }
        int []bucket = new int[nums.length+1];
        int []buMin = new int[nums.length+1];
        int []buMax = new int[nums.length+1];
        boolean []buFlag = new boolean[nums.length+1];
        double dis = (max - min)*1.0f / nums.length;
        int id;
        for(int i =0;i<nums.length;i++){
            id = (int)Math.round( (nums[i] - min) / dis);
            if(buFlag[id] == false){
                buMin[id] = buMax[id] = nums[i];
                buFlag[id] = true ;
            }else{
                buMin[id] = nums[i]<buMin[id] ? nums[i]:buMin[id];
                buMax[id] = nums[i]>buMin[id] ? nums[i]:buMax[id];
            }
        }
        int ans =Integer.MIN_VALUE;
        for(int i=0;i<bucket.length;){
            int j=i;
            while(++j < bucket.length &&  buFlag[j] == false);
            if(j >= bucket.length)
                return ans;
            if(j!= i+1 ){
                ans = buMin[j]- buMax[i]> ans ?buMin[j]- buMax[i] : ans;
                i =j;
            }else{
                ans = buMin[j]- buMax[i]> ans ?buMin[j]- buMax[i] : ans;
                i++;
            }
        }
        return ans;
    }
```

#### 8 链表相交的一系列问题

##### 8.1 leetcode 剑指Offer 52 两个链表相交的第一个公共节点

```
思路挺简单，没啥说的
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int len1 = 0, len2 = 0;
        ListNode node = headA;
        while(node!=null){
            len1++;
            node=node.next;
        }
        node = headB;
        while(node!=null){
            len2++;
            node=node.next;
        }
        

        ListNode node1 = len1 > len2 ?  headA : headB;
        ListNode node2 = len1 > len2 ?  headB : headA;
        int len3 = Math.abs(len1-len2);
        while(len3-- >0){
            node1=node1.next;
        }
        while(node1 != node2){
            node1= node1.next;
            node2= node2.next;
        }
        return node1;
    }

}
```

##### 8.2 leetcode 141 如何判断一个链表是否有环

```
function detectCycle(head) {
    let fast = head;
    let slow = head; 

    while (fast && fast.next) {
        fast = fast.next.next;
        slow = slow.next;

        if (fast == slow) {
            // 其中一个指针指向不动，另一个指针指向头
            slow = head;
            while (fast !== slow) {
                // 同时只移动一步
                slow = slow.next;
                fast = fast.next;
            }
            // 此时再次相遇，指向的那个节点就是入环节点
            return slow;
        }
    }

    return null;
}
```

##### 8.3如何判断两个有环链表是否相交, 相交则返回第一个相交节点, 不相交则返回null.

有时间回过头来写，这道题有点麻烦

#### 9、栈队列问题

##### 9.1两个栈实现一个队列

```
Stack<Integer> stack1;
    Stack<Integer> stack2;
    public CQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    public void appendTail(int value) {
        stack1.push(value);
    }
    public int deleteHead() {
        if(stack1.isEmpty())
            return -1;
        while(!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        int ans = stack2.pop();
        while(!stack2.isEmpty()){
            stack1.push(stack2.pop());
        }
        return ans;
    }
```

##### 9.2两个队列实现一个栈

```
/** Initialize your data structure here. */
    Queue<Integer> queue1;
    Queue<Integer> queue2;
    int size =0;
    public MyStack() {
        queue1 =  new LinkedList<Integer>() ;
        queue2 =  new LinkedList<Integer>() ;
        

    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        queue1.add(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        if(queue1.size() ==0)
            return -1;
        while(queue1.size() > 1){
            queue2.add(queue1.poll());
        }
        int ans = queue1.poll();
        while(!queue2.isEmpty()){
            queue1.add(queue2.poll());
        }
        return ans;
    }
    
    /** Get the top element. */
    public int top() {
        if(queue1.size() ==0)
            return -1;
        while(queue1.size() > 1){
            queue2.add(queue1.poll());
        }
        int ans = queue1.peek();
        queue2.add(queue1.poll());
        while(!queue2.isEmpty()){
            queue1.add(queue2.poll());
        }
        return ans;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue1.isEmpty();
    }
```

##### 9.3 用数组实现栈 leetcode622

```
数组实现队列
    int []arr = null;
    int curId =0 ,endId =0, curSize =0,size = 0;

    public MyCircularQueue(int k) {
        arr = new int[k];
        size =k;
    }
    
    public boolean enQueue(int value) {
        if(curSize < size){
            arr[curId% size] = value;
            curSize++;
            curId++;
            return true;
        }
        return false;
    }
    
    public boolean deQueue() {
        if(curSize<=0) return  false;
        endId ++;
        endId = endId % size;
        curSize--;
        return true;
    }
    
    public int Front() {
        if(curSize<=0) return -1;
        return arr[(endId) % size];
    }
    
    public int Rear() {
        if(curSize<=0) return -1;
        return arr[(curId-1) % size];
    
    }
    
    public boolean isEmpty() {
        return curSize==0;
    }
    
    public boolean isFull() {
        return curSize==size;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue1.isEmpty();
    }
```

##### 9.4 leetcode 641题 有点问题 。老是通不过



    /** Initialize your data structure here. Set the size of the deque to be k. */
            int []arr = null;
            int curId =0 ,endId =0, curSize =0,size = 0;
            public MyCircularDeque(int k) {
                arr = new int[k];
                size =k;
            }
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
        public boolean insertFront(int value) {
            if(curSize < size){
                arr[curId% size] = value;
                curSize++;
                curId++;
                return true;
            }
            return false;
        }
    
        /** Adds an item at the rear of Deque. Return true if the operation is successful. */
        public boolean insertLast(int value) {
            if(curSize < size){
                 endId = (endId-1) == -1 ? size-1:  endId-1;
                arr[ endId % size] = value;
                curSize++;
                return true;
            }
            return false;
        }
    
        /** Deletes an item from the front of Deque. Return true if the operation is successful. */
        public boolean deleteFront() {
            if(curSize < size){
                curId  = curId-1 == -1 ? size-1: curId-1;
                curSize--;
                return true;
            }
            return false;
        }
    
        /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
        public boolean deleteLast() {
            if(curSize<=0) return  false;
            endId ++;
            endId = endId % size;
            curSize--;
            return true;
        }
    
        /** Get the front item from the deque. */
        public int getFront() {
            if(curSize<=0) return -1;
            int id = curId-1 == -1? size-1: curId-1;
            return arr[(id) % size];
        }
    
        /** Get the last item from the deque. */
        public int getRear() {
            if(curSize<=0) return -1;
            return  arr[(endId) % size];
        }
    
        /** Checks whether the circular deque is empty or not. */
        public boolean isEmpty() {
            return curSize==0;
        }
    
        /** Checks whether the circular deque is full or not. */
        public boolean isFull() {
            return curSize==size;
        }
#### 10 设计一个getMin()函数的栈

```
/** initialize your data structure here. */
    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> minStack = new Stack<>();

public MinStack() {
}

public void push(int x) {
    if (stack1.isEmpty()) {
        minStack.push(x);
        stack1.push(x);
        return;
    }

    stack1.push(x);
    if (x < minStack.peek())
        minStack.push(x);
    else
        minStack.push(minStack.peek());
}

public void pop() {
    stack1.pop();
    minStack.pop();
}

public int top() {
    return stack1.peek();
}

public int min() {
    return minStack.peek();
}
```

#### 12-11号

#### 11、矩阵打印问题

##### (1)N*N旋转矩阵问题 面试题 01 07

```
class Solution {
    public void myswap(int [][]arr ,int tR ,int tC,int bR ,int bC ){
        int tmp = 0;
        for(int i=0;i < bC -tC ;i++ ){
            tmp = arr[bC-i][tC];
            //System.out.print ( "tmp = " + tmp  );
            arr[bC-i][tC] = arr[bR][bC-i];
            //System.out.print ( "tmp = " + arr[bC-i][tC]  );
            arr[bR][bC-i] = arr[tR+i][bC];
            //System.out.print ( "tmp = " + arr[bR][bC-i]   );
            arr[tR+i][bC] =arr[tR][ tC+i];
            //System.out.print ( "tmp = " + arr[tR+i][bC]   );
            arr[tR][ tC+i] = tmp;
           // System.out.println ( "tmp = " + arr[tR][ tC+i]   );
        }
    }
    public void rotate(int[][] matrix) {
        int N = matrix.length;
        for(int i=0;i< N ;i++){
            if(i<= N-i-1)
                myswap(matrix , i ,  i, N-i-1 ,N-i-1 );
        }
    }
}
```

##### (2）螺旋打印矩阵问题 有bug

```
public void myPrint(List<Integer> ans ,int [][] arr, int tR, int tC , int bR, int bC){
        for(int i = tC ; i< bC ;i++){
            ans.add(arr[tR][i]);
            System.out.print(arr[tR][i]);
        }
        System.out.println();
        for(int i = tR ; i< bR ;i++){
            ans.add(arr[i][bC]);
            System.out.print(arr[i][bC]);
        }System.out.println();
        for(int i = bC ; i>tC ;i--){
            ans.add(arr[bR][i]);
            System.out.print(arr[bR][i]);
        }System.out.println();
        for(int i = bR ; i>tR ;i--){
            ans.add(arr[i][tC]);
            System.out.print(arr[i][tC]);
        }System.out.println();

    }
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();
        int Row = matrix.length , Col = matrix[0].length;
        int min = Math.min(Row, Col);
        for(int i =0 ; i < min ;i++){
            myPrint(ans , matrix,i,i, Row -i -1,Col-1-i);
        }
        return ans;
    }
```

#### 12 leetcode206 反转链表

##### （1）普通情形，对所有链表反转操作

```
public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null )
            return head;
        ListNode pNext = head.next;
        ListNode pPre = null , pCur = head;
        while(pCur!=null){
            pNext = pCur.next;
            pCur.next=pPre;
            pPre= pCur;
            pCur = pNext;
        }
        return pPre;
    }
```

##### （2）进阶，leetcode 92题 对m,n之间的链表进行反转操作

增加哑结点，代码看起来很爽 c++ 版本的，代码写的真牛逼

```
public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(-1);
        dummy.next= head;
        ListNode node = dummy;
        for(int i=1;i<m;i++)
            node= node.next;
        ListNode tail=node, first = node.next; //正常链表的最后一个点，反转链表的第一个点
        ListNode pre= node.next;
        ListNode cur= node.next.next;

        for(int i=m;i<n;i++){
            ListNode next= cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        first.next =cur;
        tail.next = pre;
        return dummy.next;
    }
```

##### （3）再进阶，leetcode 25题 每K个节点之间的链表进行反转操作

爽，ac

```
public ListNode reverseKGroup(ListNode head, int k) {
        if(head == null) return head;
        int len = 0;
        ListNode dummy = new ListNode(-1);
        dummy.next=head;
        while(head!=null){
            len++;
            head=head.next;
        }
        ListNode tail = dummy;
        ListNode first = dummy.next;
        ListNode cur = dummy.next;
        ListNode pre = null;

        for(int i=0;i< len/k ;i++){
            for(int j=0;j<k;j++){
                ListNode next = cur.next;
                cur.next = pre;
                pre = cur ;
                cur = next;
            }
            first.next = cur;
            tail .next= pre;
            tail = first;
            first = cur;
            pre = first;
        }
        return dummy.next;
```

#### 13、矩阵打印 zigag问题

#### 14、 leetcode234 回文链表

##### 1、空间问题为O（N）

思路直接快慢指针到终点，压栈比对

```
public ListNode getMidNode(ListNode head){
         ListNode fast = head,slow = fast.next;
         while(fast.next!=null && fast.next.next!= null){
             fast=fast.next.next;
             slow = slow.next;
         }
         return slow;
     }
    public boolean isPalindrome(ListNode head) {
        if(head == null ||head .next == null )
            return true;
        ListNode node = getMidNode(head);
        Stack<Integer > stack1 = new Stack<>();
        while (node!=null){
            stack1.push(node.val);
            node=   node.next;
        }
        ListNode tmp = head;
        while (!stack1.isEmpty()){
            int num = stack1.pop();
            if(tmp.val != num){
                return false;
            }
            tmp=tmp.next;
        }
        return true;
    }
```

##### 2、空间O(1)

找到链表终点，反转链表，比对；再恢复链表结构

```
class Solution {
    public boolean isPalindrome(ListNode head) {
        if(head == null || head.next == null) return true;
        ListNode fast = head,slow = head;
        while(fast.next!= null  &&  fast.next.next !=null){
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode pst = slow.next;
        slow.next=null;

        ListNode pre = slow , cur = pst;
        while(cur!=null){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        pst = pre;
        ListNode phead = head;
        while(phead!=null){
            if(phead.val != pst.val)
                return false;
            pst= pst.next;
            phead=phead.next;
        }
        return  true;
    }

}
```

#### 15 leetcode 138 随机链表的复制

##### 1、哈希表

##### 2、链表复制

#### 16 程序员代码面试指南 用栈实现另外一个栈的排序

```
思路： 顺序进辅助栈，如果发现不是顺序，就吐出数据。

public static void sortStackByStack(Stack<Integer> stack){
        Stack<Integer> stack1 = new Stack<>();
        while(! stack.isEmpty()){
            int cur = stack.pop();
            while(!stack1.isEmpty() && stack1.peek() < cur){
                stack.push(stack1.pop());
            }
            stack1.push(cur);
        }
        while (!stack1.isEmpty()){
            stack.push(stack1.pop());
        }
}
```

#### 12-12号

#### 17剑指offer 第3题

https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/

##### （1）暴力哈希

```
public int findRepeatNumber(int[] nums) {
    HashMap<Integer,Integer> myhash = new HashMap<>();
    for(int i=0;i<nums.length;i++){
        if(!myhash.containsKey(nums[i])){
            myhash.put(nums[i],1);
        }else{
            return nums[i];
        }
    }
    return -1;
}
```

##### （2）遍历数组替换

```
class Solution {
    public void swap(int[] nums, int i,int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp   ;
    }
    public int findRepeatNumber(int[] nums) {
        for(int i=0;i<nums.length;i++){
            while(nums[i] != i){
                if(nums[i] == nums[nums[i]])
                    return nums[i];
                swap(nums,nums[i], i );
            }
        }
        return -1;
    }
}
```

##### （3）二分分割统计 每段数字出现的次数

#### 18 重建二叉树 这个题的解答不对  

##### （1）剑指offer 第7题 前序 中序恢复二叉树结构https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/

正确答案

```
public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        if(pre.length<=0 ||in.length<=0 || pre == null || in == null)
            return null;
        if(pre.length != in.length )
            return  null;
        TreeNode root = new TreeNode(pre[0]);
        int i=0;
        for(;i<in.length;i++) {
            if(pre[0]==in[i])
                break;
        }
        root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1,i+1), Arrays.copyOfRange(in,0,i));
        root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i+1,pre.length), Arrays.copyOfRange(in,i+1,in.length));
        return  root;
    }
```



错误答案 只能过leetcode不能过niuke

```
   Queue<Integer> queue = new ArrayDeque<>();
    public TreeNode constuctTree(int[] inorder , int left, int mid ,int right){
        if(left == right)
            return  new TreeNode(inorder[mid]);
        TreeNode node = new TreeNode(inorder[mid]);
        if(!queue.isEmpty() && left <= mid-1)
            node.left = constuctTree(  inorder, left,queue.poll(), mid-1);
        if(!queue.isEmpty()  && mid+1 <= right )
            node.right = constuctTree(  inorder, mid+1,queue.poll(), right);
        return node;
    }
   public TreeNode buildTree(int[] preorder, int[] inorder) {
   if(preorder.length<= 0 || inorder.length<= 0) return null;
    for(int i=0;i< preorder.length;i++){
        for(int j=0;j<inorder.length;j++){
            if(preorder[i] == inorder[j])
                queue.add(j);
        }
    }
    TreeNode ans =  constuctTree(  inorder, 0,queue.poll(), inorder.length-1);
    return ans;
}
```

##### （2）leetcode 106 中序后序恢复二叉树结构

https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/

```
Stack<Integer> stack = new Stack<>();
    public TreeNode constuctTree(int[] inorder , int left, int mid , int right){
        if(left == right)
            return  new TreeNode(inorder[mid]);
        TreeNode node = new TreeNode(inorder[mid]);
        if(!stack.isEmpty()  && mid+1 <= right )
            node.right = constuctTree(  inorder, mid+1,stack.pop(), right);
        if(!stack.isEmpty() && left <= mid-1)
            node.left = constuctTree(  inorder, left,stack.pop(), mid-1);    return node;
}
public TreeNode buildTree(int[] inorder, int[] postorder) {
    if(postorder.length<= 0 || inorder.length<= 0) return null;
    for(int i=0;i< postorder.length;i++){
        for(int j=0;j<inorder.length;j++){
            if(postorder[i] == inorder[j])
                stack.push(j);
        }
    }
    TreeNode ans =  constuctTree( inorder, 0,stack.pop(), inorder.length-1);
    return ans;
}
```

#### 19 二叉树的下一节点

```
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode pNode)
    {
        if(pNode == null) return  null;
        if(pNode.right != null)
        {
            TreeLinkNode myleft = pNode.right;
            while (myleft.left!=null) myleft=myleft.left;
            return myleft;
        }else{
            while(pNode.next !=null){
                if(pNode.next.left == pNode)
                    return pNode.next;
                pNode = pNode.next;
            }
            return null;
        }
    }
}
```

#### 20斐波那契数列

##### （1）leetcode 70 斐波那契问题

https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/

```
有bug
public int fib(int n) {
 	int []res = {0,1};
    if(n<2) return res[n];
    int pre = 0 ,cur =1;
    int ans = 0;
    for(int i=2;i<=n;i++){
    ans = (pre + cur)%1000000007;
    pre = cur%1000000007;
    cur = ans%1000000007 ;
    }
    return ans%1000000007;
}


```

##### （2）leetcode 70 斐波那契问题



```
public int climbStairs(int n) {
        int []arr = {1,1};
        if(n<=1)return arr[n];
        int pre = 1, ppre =1 ;
        int cur =0;
        for(int i =1;i< n ;i++){
            cur = pre +ppre;
            ppre = pre;
            pre = cur;
        }
        return cur;
    }
```



#### 21 变态青蛙跳台阶问题



归纳法，f(0) =1;f(1) =1

f(n-1) = f(0)+f(1)+f(2)+f(n-2)

f(n) = f(0)+f(1)+f(2)+f(n-1)

f(n) = 2*f(n-1)

```
public int JumpFloorII(int target) {
        if(target == 0 ||target ==1 ){
            return 1;
        }
        else {
            return 2*JumpFloorII(target-1);
        }
}
```



#### 22整数的n次方

链接 https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/



```
有bug 过不了
public double myPow(double x, int n) {
        if(n == 0) return 1.0;
        if(n==1) return x;
        if(n< 0 ){
            return 1/ (x*myPow(x , -n-1));
        }
        double res = myPow(x,n/2);
        res *=res;
        if( n%2 ==1)
            res *=x;
        return res;
}
```



#### 23旋转数组的最小值

```
class Solution {
    public int minArray(int[] numbers) {
        int low = 0, high = numbers.length-1;
        int mid;
        while(low<= high){
            mid = low + (high-low)/2;
            if(numbers[mid] > numbers[high]){
                low = mid+1;
            }else if(numbers[mid] < numbers[high]){
                high =mid;
            }else if(numbers[mid] == numbers[high]){
                high--;
            }
        }
        return numbers[low];
    }
}
```



#### 24 面试指南 单调栈

leetcode 84 最大子矩阵的大小 抽时间code一下

https://leetcode-cn.com/problems/largest-rectangle-in-histogram/

从一个数组里面找到分别离自己最近的最小的id。用一个栈维护升序，破坏了升序就清算。

#### 25面试指南 最大值减去最小值小于等于num的子数组的数量

有点难：没太懂



#### 26子集问题

三种解法

（1）用二进制遍历

```
public List<Integer> add(int[] nums ,int k ){
        List<Integer> ret = new ArrayList<>();
        int j=0;
        for(int i=k;i>0;i>>=1){
            if( (i & 1) == 1){
                ret.add(nums[j]);
            }
            j++;
        }
        return ret;
    }
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        int max = 1<<nums.length;
        for(int i=0;i<max;i++){
            List<Integer> mylist = add(nums,i);
            ans.add(mylist);
        }
        return ans;
    }


```





（2）

```
把代码先拿过来
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        res.push_back({});
        for (int i = 0;i < nums.size();i++) {
            int n = res.size();
            for (int j = 0;j < n;j++) {
                vector<int> item = res[j];
                item.push_back(nums[i]);
                res.push_back(item);
            }
        }
        return res;
    }
};
```

（3）迭代

```
这套代码在java中行不通，因为是浅拷贝

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> path;
        helper(res, nums, path, 0);
        return res;
    }

    void helper(vector<vector<int>>& res, vector<int>& nums, vector<int>& path,int k) {
        res.push_back(path);
        for (int i = k;i < nums.size();i++) {
            path.push_back(nums[i]);
            helper(res, nums, path, i + 1);
            path.pop_back();
        }
    }
};
```

#### 27 leetcode 62不同路径问题，动态规划

​        

    public int uniquePaths(int m, int n) {
            if(n<2) return 1;
            int [][]dp = new int[m][n];
            for(int i =1;i<m ;i++){
                dp[i][0] = 1;
            }
            for(int i =1;i<n ;i++){
                dp[0][i] = 1;
            }
      	for(int i =1;i<m ;i++){
            for(int j=1;j<n;j++){
                dp[i][j] = dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }





28 买卖股票的最佳时机

##### leetcode121  记录当前价格之前的最低价格

    记录当前价格之前的最低价格
    public int maxProfit(int[] prices) {
        int ans = 0, min = Integer.MAX_VALUE;
        for(int i=0;i<prices.length;i++){
            min = Math.min(min, prices[i]);
            ans  = Math.max(prices[i] - min,ans);
        }
        return  ans ;
    }
##### leetcode122 直接贪心

```
直接贪心
public int maxProfit(int[] prices) {
       int ans =0;
        for(int i=0;i<prices.length;){
            int temp = i;
            int j=i+1;
            while(j<prices.length && prices[j]>= prices[temp]  ){
                j++;
                temp++;
            }
            ans +=  prices[j-1] - prices[i];
            if(j-1 == i)
                i++;
            else i=j-1;
        }
        return ans ;
    }
```

leetcode123

我自己想的解法：两次嵌套遍历，把升序子序列加入大顶锥。然后取出不相邻的两截。

    超时了
    public int func(int[] prices,int from) {
            int ans = 0, min = Integer.MAX_VALUE;
            for(int i=from;i<prices.length;i++){
                min = Math.min(min, prices[i]);
                ans  = Math.max(prices[i] - min,ans);
            }
            return  ans ;
        }
        
    public int maxProfit(int[] prices) {
        int ans =0;
        int tmp =0;
        for(int i=0;i< prices.length;i++){
            for(int j=i+1;j<prices.length;j++){
                if(prices[j]> prices[i]){
                    tmp = prices[j] - prices[i ] + func(prices,j+1);
                    ans = Math.max(ans,tmp);
                }
            }
        }
        return ans;
    }

##### leetcode 188题 暴力递归改记忆化搜索，过不了原因还是暴力递归写的有问题

```
//public int res = 0 , tempres = 0;
    //status == 1 代表 当前已经买了一个，还要选一个， status ==0表示还没买入，即将买入
    public int process(int []prices,  int by,int sl,int status ,int k){
        if(k<=0 || by >=prices.length || sl >=prices.length || status == 1 && by==prices.length-1)
            return 0;
        int res = 0;
        for(int i= status ==0 ? by : sl;i<prices.length;i++){
            if(status ==0){//从 sl+1 到尾选一个购买
                res = Math.max(res,process(prices , i, i+1, 1 , k));
            }
            else {
                if(prices[i] >prices[by]){
                    int byBefore = process(prices , i+1,i+2,0 , k-1) + prices[i]  - prices[by];//买第i个
                    int notBy = process(prices , by,i+1,1 , k);//不买第i个
                    return  Math.max(byBefore,notBy);
                }
            }
        }
        return res;
    }
    public HashMap<String,Integer> myhash ;
    public int process1(int []prices,  int by,int sl,int status ,int k) {
        if (k <= 0 || by >= prices.length || sl >= prices.length || status == 1 && by == prices.length - 1)
            return 0;
        int res = 0;
        String mystring = String.valueOf(by) + String.valueOf(sl) + String.valueOf(status) + String.valueOf(k);
        if (myhash.containsKey(mystring)) {
            return myhash.get(mystring);
        }

​    for (int i = status == 0 ? by : sl; i < prices.length; i++) {
​        if (status == 0) {//从 sl+1 到尾选一个购买
​            int temp = 0;
​            String string = String.valueOf(i) + String.valueOf(i + 1) + String.valueOf(1) + String.valueOf(k);
​            if (!myhash.containsKey(string)) {
​                temp = process1(prices, i, i + 1, 1, k);
​                myhash.put(string, temp);
​            } else
​                temp = myhash.get(string);
​            res = Math.max(res, temp);
​        } else {
​            if (prices[i] > prices[by]) {
​                String string = String.valueOf(i + 1) + String.valueOf(i + 2) + String.valueOf(0) + String.valueOf(k - 1);
​                int byBefore = 0;
​                if (!myhash.containsKey(string)) {
​                    byBefore = process1(prices, i + 1, i + 2, 0, k - 1);
​                    myhash.put(string, byBefore);
​                } else
​                    byBefore = myhash.get(string);
​                byBefore = byBefore + prices[i] - prices[by];//买第i个

​                String string1 = String.valueOf(by) + String.valueOf(i + 1) + String.valueOf(1) + String.valueOf(k);
​                int notBy = 0;//不买第i个
​                if (!myhash.containsKey(string1)) {
​                    notBy = process1(prices, by, i + 1, 1, k);
​                    myhash.put(string1, notBy);
​                } else
​                    notBy = myhash.get(string1);
​                return Math.max(byBefore, notBy);
​            }
​        }
​    }
​    return res;

}
```

下面代码可以过

```
class Solution {
    public int process3(int []prices,  int by,int status, int k) {
        int [][][] dp = new int[prices.length+1][2][k+1];
        for(int i = prices.length-1 ; i>=0; i--){
            for(int j=1;j>=0;j--){
                for(int l = 1;l<=k ;l ++){
                    if(j == 0){
                        dp[i][j][l] = Math.max(dp[i+1][0][l] , dp[i+1][1][l]- prices[i]);
                    }else if(j==1){
                        dp[i][j][l] = Math.max(dp[i+1][1][l] , dp[i+1][0][l-1] + prices[i]);
                    }
                }
            }
        }
        return dp[0][0][k];
    }
    public int maxProfit(int k, int[] prices) {
        if(prices == null || prices.length <=1) return 0;
        int res = Integer.MIN_VALUE;
       

​    res =  process3(prices , 0,0, k) ;
​    return res;
}

}
```



递归版本如下


​     
​     class Solution {
​        public int process2(int []prices,  int by,int status, int k){
​            if(by >=prices.length  || k<=0)
​                return 0;
​            int ret1= 0,  ret2 = 0, ret3 =0;
​      ret2 = process2(prices, by + 1, status,k);//不变
​        if(status == 0) {
​            ret1 = process2(prices, by + 1, 1,k) - prices[by];//买一个
​        }
​        else if(status == 1) {
​            ret3 = process2(prices, by + 1, 0 , k-1) + prices[by];//
​        }
​    
​        return Math.max(ret2, Math.max(ret1,ret3 ));
​    }
​    
​    public int maxProfit(int k, int[] prices) {
​        if(prices == null || prices.length <=1) return 0;
​        int res = Integer.MIN_VALUE;
​        res =  process2(prices , 0,0, k) ;
​        return res;
​    }
}

##### leetcode 741题  动态规划 牛逼

```
public int process1(int []prices,  int by,int status,int fee ){
        int [][]dp = new int[prices.length+1][2];
        dp[prices.length][0] = dp[prices.length][1] = 0;
        for(int i = prices.length-1 ;i>=0 ;i--){
            for(int j = 1 ;j>=0 ; j--){
                if(j == 0){
                    dp[i][j] = Math.max(dp[i+1][1] -prices[i] , dp[i+1][0]);
                }else {
                    dp[i][j] = Math.max(dp[i+1][0] + prices[i] -fee , dp[i+1][1]);
                }
            }
        }
        return dp[0][0];
    }

public int maxProfit(int[] prices, int fee) {
    return process1(prices,0,0,fee);
}
```

##### leetcode 309题 动态规划

```
public int process(int []prices,  int by,int status  ){
        if(by >=prices.length || status == 0 && by >= prices.length-1)
            return 0;
        int ret1= 0,  ret2 = 0, ret3 =0;
        ret2 = process(prices, by + 1, status);//不变
        if(status == 0) {
            ret1 = process(prices, by + 1, 1) - prices[by];//买一个
        }
        else if(status == 1) {
            ret3 = process(prices, by + 2, 0) + prices[by];//
        }
        return Math.max(ret2, Math.max(ret1,ret3 ));
    }
    public int process1(int []prices,  int by,int status  ) {
        int [][]dp = new int[prices.length+1][2];
        dp[prices.length][0] = dp[prices.length][1] = 0;
        dp[prices.length-1][1] = prices[prices.length-1];
        for(int i = prices.length-2 ;i>=0 ;i--){
            for(int j = 1 ;j>=0 ; j--){
                if(j == 0){
                    dp[i][j] = Math.max(dp[i+1][1] -prices[i] , dp[i+1][0]);
                }else {
                    dp[i][j] = Math.max(dp[i+2][0] + prices[i] , dp[i+1][1]);
                }
            }
        }
        return dp[0][0];
    }
    public int maxProfit(int[] prices) {
        if(prices.length<=0) return 0;
        return process1(prices, 0,0);
    }


```



#### 28 有效的括号 

##### (1)leetcode 20



```
class Solution {
    public boolean isValid(String s) {
        Stack<Integer> stack = new Stack<>();
        for(int i=0;i<s.length();i++){
            if(s.charAt(i) ==  '(' || s.charAt(i) ==  '{' ||s.charAt(i) ==  '[' ){
                stack.push(i);
            }else{
                if(!stack.isEmpty() && (s.charAt(i) ==  ')' &&  s.charAt(stack.peek()) == '('  ||
                        s.charAt(i) ==  ']' &&  s.charAt(stack.peek()) == '[' ||
                s.charAt(i) ==  '}' &&  s.charAt(stack.peek()) == '{' ) ){
                    stack.pop();
                }else{
                    return false;
                }
            }
        }
        if(!stack.isEmpty())
            return  false;
        else return true;
    }
}
```

##### （2）leetcode 32

```
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        int []dp = new int[len];
        int pre=0;
        int res = 0;
        for(int i=1;i<len ; i++){
            if(s.charAt(i) == ')') {
                pre = i - dp[i - 1] - 1;
                if (pre >= 0 && s.charAt(pre) == '(')
                    dp[i] = dp[i - 1] + 2 + (pre > 0 ? dp[pre - 1] : 0);
                res = Math.max(res, dp[i]);
            }
        }
        return  res;
    }
}
```





#### 29跳表  先把代码拿过来看一下如何实现的

作者：stg
链接：https://leetcode-cn.com/problems/design-skiplist/solution/java-jing-jian-shi-xian-by-stg/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

```



    static class Node {
            int val;
            Node right, down;
            public Node(Node r, Node d, int val) {
                right = r;
                down = d;
                this.val = val;
            }
        }
        Node head = new Node(null, null, 0);
        Random rand = new Random();
        Node[] stack = new Node[64];
        public Skiplist() {
        }
        public boolean search(int target) {
            for (Node p = head; p != null; p = p.down) {
                while (p.right != null && p.right.val < target) {
                    p = p.right;
                }
                if (p.right != null && p.right.val == target) {
                    return true;
                }
            }
            return false;
        }
        public void add(int num) {
            int lv = -1;
            for (Node p = head; p != null; p = p.down) {
                while (p.right != null && p.right.val < num) {
                    p = p.right;
                }
                stack[++lv] = p;
            }
            boolean insertUp = true;
            Node downNode = null;
            while (insertUp && lv >= 0) {
                Node insert = stack[lv--];
                insert.right = new Node(insert.right, downNode, num);
                downNode = insert.right;
                insertUp = (rand.nextInt() & 1) == 0;
            }
            if (insertUp) {
                head = new Node(new Node(null, downNode, num), head, 0);
            }
        }
        public boolean erase(int num) {
            boolean exists = false;
            for (Node p = head; p != null; p = p.down) {
                while (p.right != null && p.right.val < num) {
                    p = p.right;
                }
                if (p.right != null && p.right.val <= num) {
                    exists = true;
                    p.right = p.right.right;
                }
            }
            return exists;
        }

#### 30 二叉树上总和为sum的最长路径  12.26号



	public static class Node {
			public int value;
			public Node left;
			public Node right;
	        public Node(int data) {
			this.value = data;
		}
	}
	
	public static int getMaxLength(Node head, int sum) {
		HashMap<Integer, Integer> sumMap = new HashMap<Integer, Integer>();
		sumMap.put(0, 0); // important
		return preOrder(head, sum, 0, 1, 0, sumMap);
	}
	
	public static int preOrder(Node head, int sum, int preSum, int level,
			int maxLen, HashMap<Integer, Integer> sumMap) {
		if (head == null) {
			return maxLen;
		}
		int curSum = preSum + head.value;
		if (!sumMap.containsKey(curSum)) {
			sumMap.put(curSum, level);
		}
		if (sumMap.containsKey(curSum - sum)) {
			maxLen = Math.max(level - sumMap.get(curSum - sum), maxLen);
		}
		maxLen = preOrder(head.left, sum, curSum, level + 1, maxLen, sumMap);
		maxLen = preOrder(head.right, sum, curSum, level + 1, maxLen, sumMap);
		if (level == sumMap.get(curSum)) {
			sumMap.remove(curSum);
		}
		return maxLen;
	}


#### 31二叉树是否为平衡二叉树 jzoffer 55  



    class  Info{
              int depth;
              boolean isBal;
              public Info(int depth, boolean isBal) {
              this.depth = depth;
              this.isBal = isBal;
          }
      }
    public Info solve(TreeNode root ){
          if(root== null){
              return new Info(1,true);
          }
          Info left = solve(root.left);
          if(left.isBal == false)
              return new Info(left.depth, false);
          Info right = solve(root.right);
          if(right.isBal == false)
              return new Info(right.depth,false);
          int dpt = Math.max(right.depth, left.depth )+1;
          boolean ret = right.isBal & left.isBal & Math.abs(right.depth - left.depth) <=1;
          return new Info(  dpt,  ret);
    
    }
    public boolean isBalanced(TreeNode root) {
        return solve(root).isBal;
    }


#### 32 T1是否包含T2的结构



```

```



    public static class Node {
            public int value;
            public Node left;
            public Node right;
    		public Node(int data) {
            this.value = data;
        }
    }
    public static boolean contains(Node t1, Node t2) {
        if (t2 == null) {
            return true;
        }
        if (t1 == null) {
            return false;
        }
        return check(t1, t2) || contains(t1.left, t2) || contains(t1.right, t2);
    }
    
    public static boolean check(Node h, Node t2) {
        if (t2 == null) {
            return true;
        }
        if (h == null || h.value != t2.value) {
            return false;
        }
        return check(h.left, t2.left) && check(h.right, t2.right);
    }

#### 33  leetcode1054 距离相等的条形码

```
class Solution {
    class Info{
        int cnt;
        int num;

        public Info(int cnt, int num) {
            this.cnt = cnt;
            this.num = num;
        }
    }
    public int[] rearrangeBarcodes(int[] barcodes) {
        if(barcodes.length <=1)
            return barcodes;
        PriorityQueue<Info> que = new PriorityQueue<>(new Comparator<Info>(){
            @Override
            public int compare(Info o1, Info o2){
                return -(o1.cnt-o2.cnt);
            }
        });
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i = 0;i<barcodes.length;i++){
            if(map.containsKey(barcodes[i])) {
                map.put(barcodes[i],map.get(barcodes[i])+1);
            }else{
                map.put(barcodes[i],1);
            }
        }
        for(Integer num: map.keySet()){

            int cnt = map.get(num);
            que.add(new Info(cnt, num));
        }
        int []ans = new int[barcodes.length];
        Info info1 = null, info2 =null;
        int i =0 ;
        int flag =0;
        while(i< ans.length && !que.isEmpty()){
            if(i==0 || info1.cnt<=0)
                info1 = que.poll();
            if(i==0) flag = info1.cnt> que.size()/2 ? 1 :0;
            if(i==0 || info2.cnt<=0)
                info2 =que.poll();
            while(info2.cnt > 0 && info1.cnt > 0 ) {
                ans[i] = (i & 0x01) == flag ? info2.num : info1.num;
                if( (i & 0x01 )==flag )
                    info2.cnt--;
                else
                    info1.cnt--;
                i++;
            }
        }
        while(info2.cnt-- > 0)
            ans[i++] = info2.num ;
        while(info1.cnt-- > 0)
            ans[i++] = info1.num;
        return ans;
    }
}
```

#### 34 leetcode 378  倒数最小的k个数

##### 方法1 用堆

```
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer t1, Integer t2) {
                return -(t1-t2);
            }
        });
        for (int i=0;i<matrix.length;i++){
            for(int j=0;j<matrix[0].length;j++){
                if(priorityQueue.size()<k){
                    priorityQueue.add(matrix[i][j]);
                }else {
                    priorityQueue.add(matrix[i][j]);
                    priorityQueue.poll();
                }
            }
        }
        return priorityQueue.poll();
    }
}
```



方法2 二分法

#### 35 前缀和 leetcode 1371   

##### 处理字符串有点烦

```
class Solution {
    public int getMask(char c ){
        switch (c){
            case 'a': return 1;
            case 'e': return 2;
            case 'i': return 4;
            case 'o': return 8;
            case 'u': return 16;
            default:
                return 0;
        }
    }
    public int findTheLongestSubstring(String s) {
       HashMap<Integer,Integer> map  = new HashMap<>();//词频
       char[] arr = s.toCharArray();
       map.put(0,-1);
       int m = 0,ans =0;
       for(int i=0;i<arr.length;i++){
           m ^= getMask (arr[i]);
           if(map.containsKey(m)){
               ans = Math.max(ans, i - map.get(m)   );
           }else{
               map.put(m, i);
           }
       }
       return ans;
    }
}
```



##### 前缀和 leetcode 560

```
public int subarraySum(int[] nums, int k) {
       Map<Integer, LinkedList<Integer>> map  = new HashMap<>();
        LinkedList<Integer> list = new LinkedList<>();
        list.add(-1);
        int sum = 0 ,ans =0;
        map.put(0,list);
        for(int i=0 ;i< nums .length;i++){
            sum += nums[i];
            if(map.containsKey(sum-k)){
                ans+= map.get(sum-k).size();
            }
            if(map.get(sum) == null) {
                list  = new LinkedList<>();
                list.add(i);
                map.put(sum,list);
            }else
                map.get(sum).push(i);
        }
        return  ans;
    }
```

#### 36 leetcode 根据字符串出现频率排序

```
class Solution {
    public String frequencySort(String s) {
        String ans = new String();
        HashMap<Character,Integer> map = new HashMap<>();
        char [] str = s.toCharArray();
        for(char c: str){
            map.put( c, map.getOrDefault(c,0) +1);
        }
        Set<Character> characters = map.keySet();
        List<Map.Entry<Character, Integer>> l = new ArrayList<>(map.entrySet());
        l.sort(((o1, o2) -> o2.getValue() - o1.getValue()));
        for(Map.Entry<Character, Integer> e : l){
            int t = e.getValue();
            while(t-->0){
                ans += e.getKey();
            }
        }
        return ans;
    }
}
```

#### 37 前缀树 leetcode 209 





```
class TrieNode{
    public int path;
    public int end;
    public TrieNode[ ] map;
    public TrieNode() {
        path=0;
        end =0;
        map = new TrieNode[26];
    }
}

class Trie {

    /** Initialize your data structure here. */
    TrieNode root ;
    public Trie() {
        root = new TrieNode();
    }
    /** Inserts a word into the trie. */
    public void insert(String word) {
        if(word ==null) return;
    
        char [] arr = word.toCharArray();
        TrieNode node = root;
        node.path++;
        int index =0;
        for(int i=0;i< arr.length;i++){
            index = arr[i]  - 'a';
            if(node.map[index] == null){
                node.map[index] = new TrieNode();
            }
            node.map[index].path++;
            node = node.map[index];
        }
        node.end++;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        if(word ==null) return false;
        char [] arr = word.toCharArray();
        TrieNode node = root;
        int index =0;
        for(int i=0;i< arr.length;i++){
            index = arr[i]  - 'a';
            if(node.map[index] == null){
                return false;
            }
            node = node.map[index];
        }
        return  node.end!=0 ? true : false;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        if(prefix ==null) return false;
        char [] arr = prefix.toCharArray();
        TrieNode node = root;
        int index =0;
        for(int i=0;i< arr.length;i++){
            index = arr[i]  - 'a';
            if(node.map[index] == null){
                return false;
            }
            node = node.map[index];
        }
        return node.path >0 ;
    }

}
```

#### 38 前缀树 leetcode 17 



    class TrieNode{
        public int path;
        public int end;
        public int index;
        public  TrieNode[ ] map;
        public TrieNode() {
            path=0;
            end =0;
            index = -1;
            map = new  TrieNode[26];
        }
    }
    
    class Trie {
    /** Initialize your data structure here. */
    TrieNode root ;
    public Trie() {
        root = new  TrieNode();
    }
    /** Inserts a word into the trie. */
    public void insert(String word,int idx) {
        if(word ==null) return;
    
        char [] arr = word.toCharArray();
         TrieNode node = root;
        node.path++;
        int index =0;
        for(int i=0;i< arr.length;i++){
            index = arr[i]  - 'a';
            if(node.map[index] == null){
                node.map[index] = new  TrieNode();
            }
            node.map[index].path++;
            node = node.map[index];
        }
        node.end++;
        node.index =idx;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        if(word ==null) return false;
        char [] arr = word.toCharArray();
         TrieNode node = root;
        int index =0;
        for(int i=0;i< arr.length;i++){
            index = arr[i]  - 'a';
            if(node.map[index] == null){
                return false;
            }
            node = node.map[index];
        }
        return  node.end!=0 ? true : false;
    }
    
    public int [][] slove(String []smalls, String  big){
        List<List<Integer>> list = new ArrayList<>(smalls.length);
        for (int i = 0; i < smalls.length; i++) {
            list.add(new ArrayList<>());
        }
        for (int i = 0; i < big.length(); i++) {
            String curWord = big.substring(i);
            TrieNode curTrie = root;
            char[] chars = curWord.toCharArray();
            for (int j = 0; j < chars.length; j++) {
                int charIndex = chars[j] -'a';
                if (curTrie.map[charIndex] != null) {
                    curTrie = curTrie.map[charIndex];
                }else{
                    break;
                }
                if (curTrie.end !=0) {
                    List<Integer> subList = list.get(curTrie.index);
                    subList.add(i);
                }
            }
        }
        int [][]res = new int[list.size()][];
        for(int i=0;i < res.length;i++){
            List<Integer> subList = list.get(i);
            res[i] = new int[subList.size()];
            for(int j=0;j< subList.size();j++){
                res[i][j] = subList.get(j);
            }
        }
        return  res;
    }
    }
    class Solution {
        public int[][] multiSearch(String big, String[] smalls) {
            Trie trie = new Trie();
            for(int i =0 ; i<smalls.length;i++){
                trie.insert(smalls[i],i);
            }
            return  trie.slove(smalls,big);
        }
    }

#### 39 leetcode 39 并查集

```
class Union{
    int []arr ;
    public  Union(int n){
        arr = new int[n];
        for(int i=0;i < n; i++){
            arr[i] = i;
        }
    }
    int count(){
        int sum =0;
        for(int i=0;i<arr.length;i++){
            sum += arr[i]==i ? 1 :0;
        }
        return sum;
    }
    int find(int x){
        while(x!=arr[x]){
            x =arr[x];
        }
        return x;
    }
    void unite(int x,int y){
        int rx = find(x);
        int ry = find(y);
        if(rx != ry){
            arr[ry] = rx;
        }
    }

}
class Solution {
    public int findCircleNum(int[][] isConnected) {
        Union union = new Union(isConnected.length);
        for(int i=0;i<isConnected.length;i++){
            for(int j=0;j < isConnected[0].length;j++){
                if(i!= j && isConnected[i][j]==1){
                    union.unite(i,j);
                }
            }
        }
        return union.count();
    }
}
```

### 回溯专题

#### 40 N皇后问题 秒杀 leetcode 51 52 面试题 08.12秒杀

 

    class Solution {
        public boolean isValid(char[][]board,int row, int col, int n ){
            for(int i=row-1; i>=0; i--){
                if(board[i][col] == 'Q')
                    return false;
            }
            for(int i=row-1, j=col-1; i>=0 && j>=0; i--,j-- ){
                if(board[i][j] == 'Q')
                    return false;
            }
            for(int i=row-1, j=col+1; i>=0 && j<n; i--,j++ ){
                if(board[i][j] == 'Q')
                    return false;
            }
            return true;
        }
        public void backstrace(List<List<String>> ans, int curR, int n, char[][]board){
            if(curR ==n){
                List<String> l = new ArrayList<>();
                for(int i=0;i<n;i++)
                    l.add(new String(board[i]));
                ans.add(l);
            }
            for(int i=0;i<n;i++){
                if(!isValid(board,curR,i, n)){
                   continue;
                }
                board[curR][i] = 'Q';
                backstrace(ans, curR+1, n, board);
                board[curR][i] = '.';
            }
        }
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> ans = new ArrayList();
        char[][]board =new char[n][n];
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                board[i][j] = '.';
        backstrace(ans, 0, n,board );
        return ans;
    }
    }
#### 41 滑动窗口 leetcode 76

 

```
public String minWindow(String s, String t) {  
        Map<Character,Integer> need = new HashMap<>();
        Map<Character,Integer> window = new HashMap<>();
        for(char c: t.toCharArray()){
            need.put(c,need.getOrDefault(c,0)+1);
        }
        int left = 0,right = 0 ,valid =0;
        int st=0, len = Integer.MAX_VALUE;
        while(right<s.length()){
            char c = s.charAt(right);
            right++;
            if(need.containsKey(c)){
                window.put(c, window.getOrDefault(c,0)+1);
                if(window.get(c) .equals(need.get(c)) )
                    valid++;
            }

            while(valid == need.size()){
                if(right-left < len){
                    st= left;
                    len = right- left;
                }
                char d = s.charAt(left);
                left++;
                if(need.containsKey(d)){
                    if(window.get(d) .equals(need.get(d)) )
                        valid --;
                    window.put(d, window.getOrDefault(d,0) -1);
                }
            }
        }
        return len ==Integer.MAX_VALUE?"": s.substring(st,st+len);
    } 
```

##### leetcode 567



    public boolean checkInclusion(String s1, String s2) {
           Map<Character,Integer> need = new HashMap<>();
            Map<Character,Integer> window = new HashMap<>();
            for(char c: s1.toCharArray())
                need.put(c,need.getOrDefault(c,0)+1);
            int left =0, right =0, valid =0;
            while(right<s2.length()){
                char c = s2.charAt(right);
                right++;
                if(need.containsKey(c)){
                    window.put(c,window.getOrDefault(c,0)+1);
                    if(window.get(c).equals(need.get(c)))
                        valid++;
    }
            while(right-left>=s1.length()){
                if(valid == need.size())
                    return true;
                char d = s2.charAt(left);
                left++;
    
                if(need.containsKey(d)){
                    if(window.get(d).equals(need.get(d )))
                        valid--;
                    window.put(d,window.getOrDefault(d,0)-1);
                }
    
            }
        }
        return false;
    }
##### leetcode 242  顺带解决一道异位词

    public boolean isAnagram(String s, String t) {
       Map<Character,Integer> need = new HashMap<>();
        Map<Character,Integer> window = new HashMap<>();
        for(char c: s.toCharArray())
            need.put(c, need.getOrDefault(c,0)+1);
        for(char c: t.toCharArray())
            window.put(c, window.getOrDefault(c,0)+1);
        if(need.keySet().size() != window.keySet().size())
            return false;
        for( char c :need.keySet()){
            if(!need.get(c).equals(window.get(c)))
                return false;
        }
        return true;
    }
##### leetcode 438 

```
public List<Integer> findAnagrams(String s, String p) {
        Map<Character,Integer> need = new HashMap<>();
        Map<Character,Integer> window = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for(char c: p.toCharArray())
            need.put(c,need.getOrDefault(c,0)+1);
        int left =0, right =0, valid =0;
        while(right<s.length()){
            char c = s.charAt(right);
            right++;
            if(need.containsKey(c)){
                window.put(c,window.getOrDefault(c,0)+1);
                if(window.get(c).equals(need.get(c)))
                    valid++;
            }
            while(right-left>=p.length()){
                if(valid == need.size())
                   list.add(left);
                char d = s.charAt(left);
                left++;
                if(need.containsKey(d)){
                    if(window.get(d).equals(need.get(d )))
                        valid--;
                    window.put(d,window.getOrDefault(d,0)-1);
                }
            }
        }
        return list;
    }
```



#### 42 变种 二分 leetcode  4  有精力要自己重写一下

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
            int n = nums1.length;
            int m = nums2.length;
            int left = (n + m + 1) / 2;
            int right = (n + m + 2) / 2;
            //将偶数和奇数的情况合并，如果是奇数，会求两次同样的 k 。
            return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;
        }
    
    private int getKth(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        //让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1
        if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
        if (len1 == 0) return nums2[start2 + k - 1];
    
        if (k == 1) return Math.min(nums1[start1], nums2[start2]);
    
        int i = start1 + Math.min(len1, k / 2) - 1;
        int j = start2 + Math.min(len2, k / 2) - 1;
    
        if (nums1[i] > nums2[j]) {
            return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start2 + 1));
        }
        else {
            return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start1 + 1));
        }
    }

##### 二分 leetcode 69 

```
class Solution {
    public int mySqrt(int x) {
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long) mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }
}
```

#### 43 二分搜索

##### （1）基本版二分





#### 44Leetcode 1456 

```
class Solution {
    public int maxVowels(String s, int k) {
        int ans =0;
        int aC =0;
        int i=0;
        char []arr = s.toCharArray();
        for(;i< Math.min(k,s.length());i++){
            if(arr[i]=='a'||arr[i]=='e'||arr[i]=='i'||
                    arr[i]=='o'||arr[i]=='u')
                aC++;
            ans = Math.max(ans,aC);
        }
        if(k<s.length()){
            for(;i<s.length();i++){
                if(arr[i]=='a'||arr[i]=='e'||arr[i]=='i'||
                        arr[i]=='o'||arr[i]=='u')
                    aC++;
                if(arr[i-k]=='a'||arr[i-k]=='e'||arr[i-k]=='i'||
                        arr[i-k]=='o'||arr[i-k]=='u')
                    aC--;
                ans = Math.max(ans,aC);
            }
        }
        return ans;
    }
}
```

#### leetcode 778 迪杰斯特拉

```
class Solution {
    class Point{
        int value;
        int x;
        int y;
        public Point(int x, int y, int value) {
            this.value = value;
            this.x = x;
            this.y = y;
        }
    }
    public boolean isValid(boolean [][]visit, int x, int y ,int row, int col){
        if(0<=x && x< row && 0<=y && y< col && visit[x][y] == false ){
            return true;
        }
        return false;
    }
    public int swimInWater(int[][] grid) {
        PriorityQueue<Point> queue = new PriorityQueue<>(new Comparator<Point>() {
            @Override
            public int compare(Point p1, Point p2) {
                return p1.value-p2.value;
            }
        });
        Stack<Point> stack =new Stack<>();
        int []dx = {0,1,0,-1};
        int []dy = {-1,0,1,0};
        boolean [][] visit = new boolean[grid.length][grid[0].length] ;
        boolean end = false;
        visit[0][0] = true;
        Point point = new Point(0,0,grid[0][0]);
        stack.add(point);
        queue.add(point);
        int ans = 0;
        while(!queue.isEmpty() &&end == false){
            point = queue.poll();
            ans = Math.max(ans, grid[point.x][point.y]);
            for(int i = 0 ; i<4;i++){
                int x = point.x+ dx[i];
                int y = point.y +dy[i];

                if( isValid(visit, x, y , grid.length,  grid[0].length)){
                    visit[x][y] = true;
                    if(x == grid.length-1 && y == grid[0].length-1) {
                        end = true;
                        ans = Math.max(ans, grid[x][y]);
                        break;
                    }
                    queue.add(new Point(x,y, grid[x][y] ));
                }
            }
        }
        return ans;
    }
}
```

#### 45 leetcode 300 最长上升子序的长度

定义dp[i]是以nums[i]结尾的最长上升子序的数量                                                   

```
class Solution {
    public int lengthOfLIS(int[] nums) {
        int [] dp = new int[nums.length];
         for(int i=0; i< dp.length;i++)
             dp[i] = 1;
         for(int i=0;i<nums.length;i++){
             for(int j=0;j<i;j++){
                 if(nums[i]>nums[j])
                 dp[i] = Math.max(dp[i],dp[j]+1);
             }
         }
         int ans = 0;
         for(int i=0;i<dp.length;i++){
             ans= Math.max(ans, dp[i]);
         }
         return ans;
    }
}
```

##### 俄罗斯信封套娃 leetcode 345

​    


    class Solution {
    public int lengthOfLIS(int[] nums) {
            int [] dp = new int[nums.length];
            for(int i=0; i< dp.length;i++)
                dp[i] = 1;
            for(int i=0;i<nums.length;i++){
                for(int j=0;j<i;j++){
                    if(nums[i]>nums[j])
                        dp[i] = Math.max(dp[i],dp[j]+1);
                }
            }
            int ans = 0;
            for(int i=0;i<dp.length;i++){
                ans= Math.max(ans, dp[i]);
            }
            return ans;
        }
    public int maxEnvelopes(int[][] envelopes) {
        Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] t1, int[] t2) {
                if(t1[0] != t2[0] )
                    return t1[0] - t2[0];
                else 
                    return t2[1] - t1[1];
            }
        });
        int [] lis = new int[envelopes.length];
        for(int i=0;i<lis.length;i++)
            lis[i] = envelopes[i][1];
        return lengthOfLIS(lis);
    }
    }




#### leetcode jzoffer 42 数组中的最大子串和

```
class Solution {
    public int maxSubArray(int[] nums) {
        int []dp = new int[nums.length];
        dp = nums.clone();
        int ans =dp[0];
        for(int i=1;i<dp.length;i++){
            dp[i] = Math.max(dp[i], dp[i]+dp[i-1]);
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}
```

#### leetcode 1143 最长公共子序

```
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int n1 = text1.length(), n2 = text2.length();
        int [][]dp = new int[n1][n2];
        boolean b = false;
        for(int i=0;i<n1 ;i++){
            if(text1.charAt(i) == text2.charAt(0))
                b = true;
            if(b==true)
                dp[i][0] = 1;
        }
        b=false;
        for(int i=0;i<n2 ;i++){
            if(text1.charAt(0) == text2.charAt(i))
                b = true;
            if(b==true)
                dp[0][i] = 1;
        }
        for(int i=1;i<n1;i++){
            for(int j=1;j<n2;j++){
                if(text1.charAt(i)== text2.charAt(j)){
                    dp[i][j] = dp[i-1][j-1]+1;
                }else{
                    dp[i][j] = Math.max( dp[i-1][j] , dp[i][j-1]);
                }
            }
        }
        return dp[n1-1][n2-1];
    }
}
```



#### leetcode 72 编辑距离  老套路先用暴力搜索，再改dp01月18号



    public int processdp(String word1, String word2, int l1, int l2){
            int [][]dp = new int[l1+1][l2+1];
            for(int i=0;i<l1+1;i++)
                dp[i][0] = i;
            for(int j=0;j<l2+1;j++)
                dp[0][j] = j;
            for(int i=1;i<l1+1;i++){
                for(int j =1 ;j<l2+1;j++){
                    if(word1.charAt(i-1) == word2.charAt(j-1)){
                        dp[i][j] = dp[i-1][j-1];
                    }else{
                        dp [i][j] = Math.min(  dp[i-1][j] ,Math.min(dp[i-1][j-1] ,dp[i][j-1] ) )+1;
                    }
                }
            }
            return dp[l1][l2];
    }
    
    public int process(String word1, String word2, int l1, int l2){
            //basecase
            if(l1 ==-1)
                return l2+1;
            else if(l2 == -1)
                return l1+1;
            else if(word1.charAt(l1) == word2.charAt(l2)){
                return process(word1,word2,l1-1, l2 -1);
            }
            else{
                return Math.min( process(word1,word2, l1-1, l2) + 1, //删除
                        Math.min(process(word1,word2, l1-1, l2-1) + 1,//替换
                                process(word1,word2, l1, l2-1) + 1));  //插入
            }
        }


​    
​    public int minDistance(String word1, String word2) {
​        return processdp(word1, word2,word1.length(), word2.length());
​    }


#### leetcode 516 编辑距离

```
class Solution {
    public int process1(String s , int l , int r){
        int [][]dp = new int[r][r];
        for(int i=0;i<r;i++)
            dp[i][i] =1;
        for(int i= r-2 ; i>=0;i--){
            for(int j=i+1 ;j<r;j++ ){
                if(s.charAt(i) == s.charAt(j))
                    dp[i][j] = dp[i+1][j-1]+2;
                else
                    dp[i][j] = Math.max(dp[i][j-1], dp[i+1][j]);
            }
        }
        return dp[0][r-1];
    }
    
    public int process(String s , int l , int r){
       if(l==r)
           return 1;
       if(s.charAt(l) == s.charAt(r))
           return 2+process(s,l+1,r-1);
       else
           return Math.max(process(s,l,r-1) , process(s,l+1,r) );

    }
    
    public int longestPalindromeSubseq(String s) {
        return process1(s, 0, s.length());
    }
}


```



#### leetcode 1312 [让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

```
class Solution {
    public int process(String s, int l , int r){
        int [][]dp = new int[r][r];
        for(int i=r-2;i>=0;i--){
            for(int j=i+1;j<r;j++){
                if(s.charAt(i)==s.charAt(j)){
                    dp[i][j] = dp[i+1][j-1];
                }else {
                    dp[i][j] = Math.min(dp[i+1][j], dp[i][j-1])+1;
                }
            }
        }
        return dp[0][r-1];
    }
    public int minInsertions(String s) {
        return process(s,0,s.length());
    }
}
```



#### leetcode 10  正则表达式  jzoffer 19



    class Solution {//isMatch表示s字符匹配到i位置， p字符串匹配到j位置 m ,n 是s 和p 的长度
        public boolean isMatch(char[] s , char[] p , int i, int j ,int m , int n){
            if(j==n){//j到了末尾，只有i到了末尾返回true
                return i==m;
            }
            if(i==m){//i到了末尾，只有 *a*a *号和字符重复交错才返回true
                if((n - j) % 2 == 1)//如果这个时候是还剩奇数个字符，肯定不满足交错，直接返回false
                    return false;
                for(;j+1<n;j+=2){//每次跳2格，如果不等于*号直接返回false
                    if(p[j+1] != '*')
                        return false;
                }
                return true;
            }
    		if(s[i] == p[j] || p[j] =='.'){//可以匹配上
            if(j<n-1 && p[j+1] == '*'){//如果p[j+1]是*，可以跳过i字符， 也可以把j和j+1 丢掉
                return isMatch(s,p, i+1, j,m,n) || isMatch(s,p, i,j+2, m,n);
            }else {//如果p[j+1]不是*，只能i+1,j+1
                return isMatch(s,p,i+1,j+1,m,n);
            }
        }else{
            if(j<n-1 && p[j+1] == '*') {//没有匹配上，就只能丢掉j和j+1
                return isMatch(s,p, i, j+2,m,n);
            }else{//否则返回 false
                return false;
            }
        }
    }//会哥直接写，（第一道题可以暴力排序，）
    public boolean isMatch(String s, String p) {
        char[] cs = s.toCharArray();
        char[] cp = p.toCharArray();
        return  isMatch (cs, cp, 0,0, s.length(), p.length());
    }
    }






#### leetcode 887 高楼[887. 鸡蛋掉落  ](https://leetcode-cn.com/problems/super-egg-drop/)  这道题比较难，有时间要好好体会





    class Solution {
        Map<String, Integer> map;
        public  int process(int K, int N){
            if (N == 0 || N == 1 || K == 1) {
                return N;
            }
            String s =  Integer.toString(K)+'_'+Integer.toString(N);
            if(map.containsKey(s))
                return map.get(s);
            int res = N;
            int l =1, h = N;
    
    	while(l<=h){
            int mid = (l+h)/2;
            int brok = process(K-1, mid-1);
            int not_brok = process(K, N-mid);
            if(brok > not_brok) {
                h = mid -1;
                res = Math.min(res, brok+1);
            }else{
                l = mid+1;
                res = Math.min(res, not_brok+1);
            }
        }
        map.put(s, res);7
        return res;
    }
    public int superEggDrop(int K, int N) {
        map = new HashMap<>();
        return  process(K,N);
    }
    }


#### leetcode 312 打气球  牛逼 [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)

```
class Solution {
    public int maxCoins(int[] nums) {
        int []point = new int[nums.length+2];
        for (int i = 0; i < nums.length; i++) {
            point[i+1] = nums[i];
        }
        int [][]dp = new int[nums.length+2][ nums.length+2];
        point[0] = point[nums.length+1] = 1;
        int n = nums.length+2;
        for (int i = n-1; i >=0 ; i--) {
            for (int j = i+1; j < n; j++) {
                for (int k = i+1; k < j; k++) {
                    dp[i][j] = Math.max(dp[i][j] , dp[i][k] + dp[k][j] + point[i]*point[j] * point[k]);
                }
            }
        }
        return dp[0][n-1];
    }
}
```

#### leetcode [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

```
class Solution {
    public boolean canPartition(int[] nums) {
       int sum =0;
        for (int i = 0; i < nums.length; i++) {
            sum+= nums[i];
        }
        if(sum%2==1)    return false;
        int tar = sum/2;
        boolean [][]dp = new boolean[nums.length+1][tar+1];
        for(int i=0;i<nums.length;i++)
            dp[i][0] = true;
        for (int i = 1; i <= nums.length; i++) {
            for (int j = 1; j <= tar; j++) {
                if(j<nums[i-1]){
                    dp[i][j] = dp[i-1][j];
                }else{
                    dp[i][j] = dp[i-1][j] ||  dp[i-1][j-nums[i-1]];
                }
            }
        }
        return dp[nums.length][tar];
    }
}
```



#### [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)



```


class Solution {
    public int change(int amount, int[] coins) {
        int n =coins.length;
        int [][]dp = new int[n+1][ amount+1];
        for(int i=0;i<=amount;i++)
            dp[0][i] = 0;
        for(int i=0;i<=coins.length;i++)
            dp[i][0] =1;
        for(int i=1;i<=n;i++){
            for (int j = 1; j <= amount; j++) {
                if(j< coins[i-1]){
                    dp[i][j] = dp[i-1][j];
                }else{
                    dp[i][j] = dp[i-1][j]+dp[i][j-coins[i-1]];
                }
            }
        }
        return dp[n][amount];
    }
}
```

#### leetcode 322 零钱兑换问题  这题没a出来 01月19号

```
 public int coinChange(int[] coins, int amount) {
        int[][] dp=new int[coins.length+1][amount+1];
        //BaseCase条件
        for (int[] n:dp) {
            Arrays.fill(n,amount+1);//最大是12张纸币
        }
        for (int i = 0; i <=coins.length ; i++) {
            dp[i][0]=0;
        }
        //套模板
        for (int i = 1; i <=coins.length; i++) {
            for (int j = 1; j <=amount; j++) {
                if (j>=coins[i-1] && dp[i][j-coins[i-1]]!=amount+1){
                    //这里注意如果选择第i个硬币,那么就是dp[i][j-coins[i-1]]+1
                    dp[i][j] = Math.min (dp[i-1][j],dp[i][j-coins[i-1]]+1);
                }
                else {
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        if (dp[coins.length][amount]==amount+1){
            dp[coins.length][amount]=-1;
        }
        
        return dp[coins.length][amount];
}
```





#### leetcode 494 直接暴力搜索居然过了  





```
class Solution {
    public int process(int[] nums, int S, int i , int sum){
        if(i == nums.length  ) {
            if (S == sum)
                return 1;
            else
                return 0;
        }
        return process(nums,S, i+1, sum+nums[i]) + process(nums,S, i+1, sum-nums[i]);
    }
    public int findTargetSumWays(int[] nums, int S) {
        return process(nums,S, 0,0);
    }
}
```



#### leetcode 474 dp  没a出来





```


public void preProcess(String[] strs,  int []cnt_0 ,int []cnt_1){
        int k=0;
        for(String s : strs){
            for(int i = 0 ; i < s.length();i++){
                if(s.charAt(i) == '0') cnt_0[k]++;
                else cnt_1[k]++;
            }
            k++;
        }
    }
    public int process(int []cnt_0 ,int []cnt_1 , int m , int n ,int i, int curm,int curn){
        if(i == cnt_0.length)
            return 0;
```







        if(curm + cnt_0[i] > m || curn + cnt_1[i] > n)//不满足
            return process(cnt_0 ,cnt_1 ,  m ,  n , i+1,  curm, curn);
        else //选或不选
            return Math.max(process(cnt_0 ,cnt_1 ,  m ,  n , i+1,  curm, curn),
                    process(cnt_0 ,cnt_1 ,  m ,  n , i+1,  curm + cnt_0[i], curn + cnt_1[i]) + 1);
    }
    public int processdp(int []cnt_0 ,int []cnt_1 , int m , int n ){
        int [][][]dp = new int[cnt_0.length+1][m+1][n+1];
        for (int j = 0; j < m+1; j++) {
            for (int k = 0; k < n+1; k++) {
                dp[cnt_0.length][j][k] = 0;//base case
            }
        }
        for (int j = cnt_0.length-1; j >=0 ; j--) {
            for (int k = 0; k < m+1; k++) {
                for (int l = 0; l < n+1; l++) {
                    if(k + cnt_0[j] > m || l + cnt_1[j] > n)
                        dp[j][k][l] = dp [j+1][k][l];
                    else
                        dp[j][k][l] = Math.max(dp[j+1][k][l], dp[j+1][k+cnt_0[j]][l+ cnt_1[j]] +1);
                }
            }
        }
        return dp[0][m][n];
    }
    
    public int findMaxForm(String[] strs, int m, int n) {
        int []cnt_0 = new int[strs.length];
        int []cnt_1 = new int[strs.length];
        preProcess(strs,  cnt_0 ,cnt_1);
    
        return processdp(cnt_0, cnt_1, m, n);
    
    }
#### leetcode 739  单调栈 



```
class Solution {
     public int[] dailyTemperatures(int[] T) {
        Stack<Integer> stack = new Stack<>();
        int [] arr = new int[T.length];
        for (int i = T.length-1; i >=0 ; i--) {
            while(!stack.isEmpty() && T[stack.peek()] <= T[i]){
                stack.pop();
            }
            arr[i] = stack.isEmpty()? 0 : stack.peek() - i;
            stack.push(i);
        }
        return arr;
    }
}
```





#### leetcode 234 判断链表是不是回文



    class Solution {
    ListNode l;
     public boolean traverse(ListNode r){
         if(r == null)
             return true;
         boolean res = traverse(r.next);
         res = res && (r.val == l.val);
         l=l.next;
         return res;
     }
    public boolean isPalindrome(ListNode head) { 
         if(head == null ||head .next == null )
            return true;
         l = head;
         return  traverse(head);
    }
    }



#### leetcode 78 排列组合子集问题

```
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        if (nums.empty()) return{ {} };
        int n = nums.back();
        nums.pop_back();
        vector<vector<int>> res = subsets(nums);
        int size = res.size();
        for (int i = 0; i < size; i++) {
            res.push_back(res[i]);
            res.back().push_back(n);
        }
        return res;
    }
};
```



#### leetcode 77 组合问题



    List<List<Integer>> ans = new ArrayList<>();
        public void backtrack(int n, List<Integer> list  ,int st , int k){
            if(list.size() == k){
                List<Integer> list1 = new ArrayList<>();
                for(int i:list)
                    list1.add(i);
                ans.add( list1);
                return;
            }
    
    	for(int i=st; i <=n ;i++){
            list.add(i);
            backtrack(n,list,i+1, k);
            list.remove(list.size()-1);
        }
    }
    public List<List<Integer>> combine(int n, int k) {
        if(n<=0 || k<=0) return ans;
        List<Integer> list = new ArrayList<>();
        backtrack(n, list, 1, k);
        return ans;
    }


#### leetcode 46 全排列问题

```


List<List<Integer>> ans = new ArrayList<>();
    public void backtrack(int[] nums, List<Integer> list , boolean []visit){
        if(list.size() == nums.length){
            List<Integer> list1 = new ArrayList<>();
            for(int i:list)
                list1.add(i);
            ans.add( list1);
            return;
        }
        for(int i=0; i < nums.length ;i++){
            if(visit[i])
                continue;
            visit[i] = true;
            list.add(nums[i]);
            backtrack(nums,list,visit);
            list.remove(list.size()-1);
            visit[i] = false;
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        if(nums.length<=0 ) return ans;
        boolean []visit = new boolean[nums.length];
        List<Integer> list = new ArrayList<>();
        backtrack(nums,list,visit);
        return ans;
    }


```



#### leetcode 37 解数独 


​        
​    class Solution {
​        public int m, n ;
​        public boolean isValid(char [][]board ,int rol, int col , char c){
​            for (int i = 0; i < board.length; i++) {
​                if( i!=rol &&  board[i][col] == c)
​                    return false;
​            }
​            for (int j = 0; j < board[0].length; j++) {
​                if( j!=col &&  board[rol][j] == c)
​                    return false;
​            }
​            for (int i = 0; i < 9; i++) {
​                if(board[(rol/3)*3 +i/3 ][(col/3)*3 +i%3] == c)
​                    return false;
​            }
​            return true;
​        }
​        public boolean backtrack(char [][]board ,int i, int j){
​            if(j==n){
​                return backtrack(board, i+1, 0);
​            }
​            if(i== m)
​                return true;
​            if(board[i][j] != '.')
​                return backtrack(board,i,j+1);
​    		for(char c = '1' ; c<= '9' ;c++ ){
​                if(!isValid(board , i, j ,c))
​                    continue;
​                board[i][j] = c;
​                if( backtrack(board, i,j+1))
​                    return true;
​                board[i][j] = '.';
​        	}
​        	return false;
​    	}
​    	public void solveSudoku(char[][] board) {
​              m = board.length;
​              n = board[0].length;
​              backtrack(board, 0, 0);
​        }
​    }
#### leetcode 22 生成括号 

```
class Solution {
    List<String> ans ;
    public void backtrack(String s , int n, int l ,int r ){
        if(r<0 || l<0)
            return;
        if(r< l )
            return;
        if(l==0 && r==0)
            ans.add(new String(s));
        s = s+ '(';
        backtrack(s, n, l-1,r );
        s = s.substring(0,s.length() - 1);
```



            s = s+')';
            backtrack(s, n, l, r-1);
            s = s.substring(0, s.length()-1);
    
        }
        public List<String> generateParenthesis(int n) {
            ans = new ArrayList<>();
            String s = new String();
            backtrack(s  , n  , n , n);
            return ans;
        }
    }                                                                                            





#### leetcode 319 电灯开关问题 只有平方数才能被最后留下



```
class Solution {
    public int bulbSwitch(int n) {
        return (int) Math.sqrt(n);
    }
}
```



#### leetcode  877 石子游戏  谁拿谁就赢

```
class Solution {
    public boolean stoneGame(int[] piles) {
        return true;
    }
}
```

leetcode 773 bfs穷尽搜索。用java有点写的不优雅



```
class Solution {
    public String retKey(ArrayList<Integer> arr){
        String s = new String();
        for(int a :arr) s = s +a;
        return s;
    }
    class Info{
        ArrayList<Integer> arr;
        int zid;
        public Info(ArrayList<Integer> arr, int zid) {
            this.arr = arr;
            this.zid = zid;
        }
    }
    public ArrayList<Integer> createArrlist(int[] board, int i , int j) {
        if(i==-1  && j==-1) {
            ArrayList<Integer> arr = new ArrayList<>();
            for(int k=0;k<board.length ;k++)
                arr.add(board[k]);
            return arr;
        }
        int t = board[i];
        board[i]= board[j];
        board[j]  = t;
        ArrayList<Integer> arr = new ArrayList<>();
        for(int k=0;k<board.length ;k++)
            arr.add(board[k]);
        t = board[i];
        board[i]= board[j];
        board[j]  = t;
        return arr;
    }
    public int slidingPuzzle(int[][] board) {
```



        int [][]map = { {1,3}, {0,2,4}, {1,5}, {0,4}, {1,3,5}, {2,4}};
        int[] board1 = new int[board.length * board[0].length];
        int id = -1;
        for (int i = 0; i < board.length; i++){
            for (int j = 0; j < board[0].length; j++) {
                board1[i * 3 + j] = board[i][j];
                if (board1[i * 3 + j] == 0)
                    id = i * 3 + j;
            }
        }
        HashSet<String> set =new HashSet<>();
        String target = "123450";
    
        ArrayList<Integer> arr1 = createArrlist(board1,-1, -1);
        String str2 = retKey(arr1);
        set.add(str2);
        Queue<Info> que = new LinkedList<>();
        que.add( new Info( arr1 , id));
    
        int ans = 0;
        while(!que.isEmpty()){
            int size = que.size();
            for(int i=0;i<size;i++) {
                Info info = que.poll();
                String str = retKey(info.arr);
                if(str.equals(target))
                    return ans;
                for(int l=0;l<board1.length;l++)
                    board1[l] = info.arr.get(l);
                for (int j = 0; j < map[info.zid].length; j++) {
                    ArrayList<Integer> arr = createArrlist(board1, map[info.zid][j], info.zid);
                    String str1 = retKey(arr);
                    if(!set.contains(str1)) {
                        que.add(new Info(arr, map[info.zid][j]));
                        set.add(str1);
                    }
                }
            }
            ans++;
        }
        return -1;
    }
    }


#### 剑指offer 57 双指针问题

```

class Solution {
    public int[] twoSum(int[] nums, int target) {
        Arrays.sort(nums);
        int l = 0,r = nums.length-1;
        while(l<r){
            int sum = nums[l]+nums[r];
            if(sum> target){
                r--;
            }else if(sum< target){
                l++;
            }else{
                return new int[]{nums[l] , nums[r]};
            }
        }
        return null;
    }
}
```



#### leetcode 15题 三数之和 

```
public List<List<Integer>> twoSum(int []nums, int st,int target){
        List<List<Integer>> ans = new ArrayList<>();
        int l = st,r = nums.length-1;
        while(l<r){
            int lo = nums[l], hi = nums[r];
            int sum = nums[l]+nums[r];
            if(sum> target){
                r--;
            }else if(sum< target){
                l++;
            }else{
                //ans.add( new ArrayList<Integer>(nums[l], nums[r]));
                ArrayList<Integer> arr1 = new ArrayList<>();
                arr1.add(nums[l]);arr1.add(nums[r]);
                ans.add(arr1);
                while(l<r && hi == nums[r]) r--;
                while(l<r && lo == nums[l]) l++;

​        }
​    }
​    return  ans;
}
public List<List<Integer>> threeSum(int[] nums) {
​    List<List<Integer>> ans = new ArrayList<>();
​    Arrays.sort(nums);
​    for(int i=0;i<nums.length;){
​        int n = nums[i];
​        List<List<Integer>> ret = twoSum(nums,i+1, -nums[i]);
​        if(ret.size()>0){
​           for( List<Integer> list: ret){
​               list.add(0,nums[i]);
​               ans.add(list);
​           }
​        }
​        while(i<nums.length && nums[i] == n) i++;
​    }
​    return ans;
}
```

#### leetcode 18 四数之和套娃问题在labuladong的算法小抄里面介绍了一种n数之和的方法，递归，由于语言特性，使用c++比java方便很多。有时间要亲自书写一遍。





     public List<List<Integer>> twoSum(int []nums, int st,int target){
            List<List<Integer>> ans = new ArrayList<>();
            int l = st,r = nums.length-1;
            while(l<r){
                int lo = nums[l], hi = nums[r];
                int sum = nums[l]+nums[r];
                if(sum> target){
                    r--;
                }else if(sum< target){
                    l++;
                }else{
                    //ans.add( new ArrayList<Integer>(nums[l], nums[r]));
                    ArrayList<Integer> arr1 = new ArrayList<>();
                    arr1.add(nums[l]);arr1.add(nums[r]);
                    ans.add(arr1);
                    while(l<r && hi == nums[r]) r--;
                    while(l<r && lo == nums[l]) l++;
    
    		}
        }
        return  ans;
    }
    public List<List<Integer>> threeSum(int[] nums ,int st,int target) {
        List<List<Integer>> ans = new ArrayList<>();
        //Arrays.sort(nums);
        for(int i=st;i<nums.length;){
            int n = nums[i];
            List<List<Integer>> ret = twoSum(nums,i+1, target-nums[i]);
            if(ret.size()>0){
               for( List<Integer> list: ret){
                   list.add(0,nums[i]);
                   ans.add(list);
               }
            }
            while(i<nums.length && nums[i] == n) i++;
        }
        return ans;
    }
    
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        for(int i=0;i<nums.length;){
            int n = nums[i];
            List<List<Integer>> ret = threeSum(nums,i+1, target-nums[i] );
            if(ret.size()>0){
                for( List<Integer> list: ret){
                    list.add(0,nums[i]);
                    ans.add(list);
                }
            }
            while(i<nums.length && nums[i] == n) i++;
        }
        return ans;
    }
#### leetcode 224 计算器表达式求值







```
class Solution {
    public boolean isdigit(char c ){
        return  '0'<=c  && c<='9';
    }
    public int [] process(char [] arr, int st){
        Stack<Integer> stack = new Stack<>();
        int num = 0 , pre = 0 , i=st;
        char sign = '+';
        int []ret = null;
        for(;i< arr.length;i++){
            if(arr[i] == '('){
                ret = process(arr, Math.max(st, i) +1); // diyige 1怎么回事？
                i= ret[1]+1 > arr.length -1? arr.length-1 :ret[1]+1 ;//已经处理完了
                num = ret[0];
            }
		
		if(i<arr.length && isdigit(arr[i]))
            num = num*10 + (arr[i] - '0');
        if(!isdigit(arr[i])&& arr[i] != ' ' || i >= arr.length-1){
            switch (sign){
                case '+':
                    stack.push(num);
                    pre = num;
                    break;
                case '-':
                    stack.push(-num);
                    pre = -num;
                    break;
                case '*':
                    stack.pop();
                    stack.push(pre* num);
                    pre = pre*num;
                    break;
                case '/':
                    stack.pop();
                    stack.push(pre/ num);
                    pre = pre/num;
                    break;
            }
            if(i< arr.length)sign = arr[i];
            num =0;
        }
        if(i< arr.length && arr[i] == ')')
            break;
    }
    int res =0;
    while (!stack.isEmpty())
        res += stack.pop();
    return new int[]{res , i};
}
public int calculate(String s) {
    char []arr = s.toCharArray();
    return process(arr, 0)[0];
}

}
```





#### leetcode 204 求区间内的素数个数 超时了

```
class Solution {
    public boolean isPrimes(int n){
        int id = (int)Math.sqrt(n);
        for(int i=2;i <= id;i++){
            if (n / i == 0)
                return false;
        }
        return true;
    }
    public int countPrimes(int n) {
        boolean [] b = new boolean[n];
        Arrays.fill(b,true);
        for(int i=2 ;i<n;i++){
            if(isPrimes(i))
                for(int j = 2*i;j<n;j+=i)
                    b[j]= false;
        }
        int cnt = 0;
        for (int i = 2; i < b.length; i++) {
            if(b[i])
                cnt++;
        }
        return cnt;
    }
}
```

 

优化后 

```
public int countPrimes(int n) {
        boolean [] b = new boolean[n];
        Arrays.fill(b,true);
        for(int i=2 ;i*i<n;i++){
            if(b[i])
                for(int j = i*i;j<n;j+=i)
                    b[j]= false;
        }
        int cnt = 0;
        for (int i = 2; i < b.length; i++) {
            if(b[i])
                cnt++;
        }
        return cnt;
    }
```

#### leetcode 1755 双向dfs 加 双指针

```
class Solution {
    List<Integer> list1 = new ArrayList<>();
    List<Integer> list2 = new ArrayList<>();
    public void process(List<Integer> list, int []nums,int sum,int i, int end){
        if(i == end){
            list.add(sum);
            return;
        }
        process(list,nums, sum, i+1, end);
        process(list,nums, sum + nums[i], i+1, end);
    }
    public int minAbsDifference(int[] nums, int goal) {
        process(list1,nums,  0, 0, nums.length/2);
        process(list2,nums, 0, nums.length/2, nums.length);
        Collections.sort(list1);
        Collections.sort(list2);
        int ans = Integer.MAX_VALUE;
        for(int a : list1){
            ans = Math.min(ans , Math.abs(a - goal));
            if(ans == 0) return ans;
        }
        for(int a : list2){
            ans = Math.min(ans , Math.abs(a - goal));
            if(ans == 0) return ans;
        }
        int i=0 ,j = list2.size() -1;
        while (i< list1.size()){
            while (j >=0  && list1.get(i) + list2.get(j) > goal ){
                ans = Math.min(ans , Math.abs(list1.get(i) + list2.get(j) - goal));
                if(j==0) break;
                else j--;
            }
            ans = Math.min(ans , Math.abs(list1.get(i) + list2.get(j) - goal));
            i++;
        }
        return ans ;
    }
}
```



#### jz offer 第六题 二分 细节是魔鬼 需要自测数据集想出各种场景



    public int getMinOrder(int [] array, int l , int r){
        int minNum =array[l];
        for(int i=l+1;i<=r ;i++){
            minNum = Math.min(minNum, array[i]);
        }
        return  minNum;
    }
    public int minNumberInRotateArray(int [] array) {
        if(array.length<=0) return 0;
        int l = 0, r = array.length-1;
        while( array[r] <= array[l]){
            int mid = (l +r )/2;
            if(r-1 == l)
                return array[r];
            if(array[mid] == array[l] && array[mid] == array[r]) {
                return  getMinOrder(array, l , r);
            }
            else if(array[mid] <= array[r]){
                r = mid;
            }else if(array[mid] >=  array[l]){
                l = mid;
            }
        }
        return array[r];
    }
#### js offer 66 题 手写bfs 搜索





    class Info{
                int rol;
                int col;
                Info(int r ,int c){
                    this.rol = r;
                    this.col = c;
                }
            }
            public boolean isValid(boolean [][] visit , int rols, int cols ,int i ,int j ,int threshold){
    		if( i>=0 && i< rols && j>=0 && j<cols && visit[i][j] == false ){
                int sum = 0;
                sum = sum + i %10;
                while(i / 10 > 0){
                    i = i/10;
                    sum = sum + i %10;
                }
                sum = sum + j %10;
                while(j / 10 > 0){
                    j = j/10;
                    sum = sum + j %10;
                }
                return sum <= threshold;
            }
            return false;
        }
        public int movingCount(int threshold, int rows, int cols)
        {
            Queue<Info> queue = new ArrayDeque<>();
            if(rows<0 || cols <0 || threshold <0) return 0;
            int ans = 1;
            boolean [][] visit = new boolean[rows][cols];
            visit[0][0] = true;
            queue.add(new Info(0,0));
            int []ofx ={-1, 0, 0,1};
            int []ofy ={0, -1, 1,0};
    
            while(!queue.isEmpty()){
                int sz = queue.size();
                for(int i=0;i<sz;i++){
                    Info  t = queue.poll();
                    for(int j=0;j<4;j++){
                        int x = t.rol + ofx[j];
                        int y = t.col+ ofy[j];
                        if(isValid(visit,rows, cols, x ,  y, threshold)){
                            ans++;
                            visit[x][y] = true;
                            queue.add(new Info( x ,y   ) );
                        }
                    }
                }
            }
            return ans;
        }
#### jzoffer 65 dfs搜索




```
  public class Solution {
​    public boolean dfs(boolean [][]visit, char[] str,int cur, char[] matrix, int i,int j
​            ,int rows,int cols){
​        if(matrix[i*cols + j] != str[cur])
​            return false;
​        if(cur == str.length-1)  return true;
​        int []ofx= {0,-1,0,1};
​        int []ofy= {-1,0,1,0};
​        for(int k=0;k<4;k++){
​            int x = i+ofx[k];
​            int y = j+ofy[k];
​    
​            if(x>=0 && x< rows && y>=0 && y< cols && visit[x][y] == false){
​                visit[x][y] = true;
​                if(dfs(visit, str, cur+1, matrix, x,y ,rows, cols))
​                    return true;
​                visit[x][y] = false;
​            }
​        }
​        return false;
​    
    }
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        boolean [][]visit = new boolean[rows][cols];
        for(int i=0;i<rows ;i++){
            for(int j=0;j<cols;j++){
                visit[i][j] = true;
                if(dfs(visit, str, 0, matrix, i,j ,rows, cols))
                    return true;
                visit[i][j] = false;
            }
        }
        return false;
    }
}
```



#### jzoffer 11  二进制中1的个数

只能处理正数不能处理负数

```
public int NumberOf1(int n) {
        int num =1;
        int ans =0;
        for(int i=0;i< 32;i++){
            if((n & num) > 0)
                ans ++;
            num = num<<1;
        }
        return ans;
    }
```

```
这个 正数负数通吃
public class Solution {
    public int NumberOf1(int n) {
        int ans =0;
        while(n!=0){
           ans++;
           n = (n-1) & n;
        }
        return ans;
    }
}
```

#### jzoffer 12 整数的指数次方

基础解法

```
public  double mypow(double base, int exponent){
        double ans = 1;
        for(int i=0;i<exponent;i++){
            ans = ans * base;
        }
        return  ans;
    }
    public  double Power(double base, int exponent) {
        if(base <= 1e-6 && base >= -1e-6)
            return 0;
        if(exponent == 0)
            return 1.0;
        if(exponent > 0){
            return mypow(base,exponent);
        }else{
            return mypow(1.0/base, -exponent );
        }
    }
```

优化 

```
public  double mypow(double base, int exponent){
        double ans = 1;
        double num = base;
        while(exponent>0){
            if(exponent %2 ==1){
                ans = ans * num;
            }
            num = num*num;
            exponent=exponent>>1;
        }
        return  ans;
    }
```

#### jzoffer 13 题 调整数组奇数位于偶数前

```
public class Solution {
    public void swap(int []array , int i ,int j){
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    public void reOrderArray(int [] array) {
        if (array == null || array.length <= 0) {
            return;
        }
        int ji =array.length -1, ou =array.length-1, r = array.length-1;
        while( ji >=0 && array[ji] % 2 == 0)
            ji--;
        while( ji >=0 && ou >=0 ){
            ou = ji-1;
            while(ou>=0 && array[ou] % 2 == 1)
                ou--;
            if(ou<0) break;
            for (int i = ou; i < ji; i++) {
                swap(array,i,i+1);
            }
            ji --;
        }
    }
}
```



#### jzoffer 14 输出链表倒数第k个节点



```
public ListNode FindKthToTail(ListNode head,int k) {
        ListNode node = head;
        while (node!=null && k != 0){
            k--;
            node=node.next;
        }
        if(k>0) return null;
        while(node!=null){
            head=head.next;
            node=node.next;
        }
        return head;
    }
```



jzoffer 15 合并两个升序链表

```
public static void process(ListNode head, ListNode list1,ListNode list2) {
        if(list1 == null) {
            head.next = list2;
            return;
        }else if(list2 == null){
            head.next = list1;
            return;
        }
        if(list1.val < list2.val) {
            head.next = list1;
            process(head.next, list1.next, list2);
        }
        else{
            head.next = list2;
            process(head.next,list1,list2.next);
        }
    }
    public static ListNode Merge(ListNode list1,ListNode list2) {
        if(list1 == null )  return list2;
        else if(list2 == null) return list1;
        ListNode ans = null;
        if(list1.val < list2.val) {
            ans = list1;
            process(ans, list1.next, list2);
        }
        else{
            ans = list2;
            process(ans,list1,list2.next);
        }
        return  ans ;
    }
```





#### jzoffer 19  顺时针打印矩阵

```
public  void printMatrix(int [][] matrix, int r,int c,int row, int col,ArrayList<Integer> ans ) {
        if(matrix==null || matrix.length<=0) return;
        int rows= matrix.length, cols = matrix[0].length;
        if(r == row)
            for (int i = c; i <=col ; i++)
                ans.add(matrix[r][i]);
        else if(c == col)
            for (int i = r; i <=row ; i++)
                ans.add(matrix[i][c]);
        else{
            for (int i = c; i < col; i++) {
                ans.add(matrix[r][i]);
            }
            for (int i = r; i < row; i++) {
                ans.add(matrix[i][col]);
            }for (int i = col; i > c; i--) {
                ans.add(matrix[row][i]);
            }for (int i = row; i > r; i--) {
                ans.add(matrix[i][c]);
            }
        }
```



    }
    public   ArrayList<Integer> printMatrix(int [][] matrix) {
        if(matrix==null || matrix.length<=0) return null;
        int rows= matrix.length-1, cols = matrix[0].length-1;
        int i=0,j=0;
        ArrayList<Integer> ans = new ArrayList<>();
        while(i<= rows && j<=cols){
            printMatrix(matrix, i,j,rows,cols,ans);
            i++;j++;rows--;cols--;
        }
        return ans;
    }




### 二叉树专题kill

#### 1.面试指南 162

```
package mianshizhinan162;
import java.util.*;
class Node {
    int val;
    Node left;
    Node right;
    public Node(int val) {
        this.val = val;
    }
}

public class Solution
{
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        Node[] map = new Node[len + 1];
        int root = in.nextInt();
        map[root] = getNode(root);
        for (int i = 2; i <= len; i++) {
            int parent = in.nextInt();
            int l = in.nextInt();
            int r = in.nextInt();
            map[l] = getNode(l);
            map[r] = getNode(r);
            map[parent].left = map[l];
            map[parent].right = map[r];
        }
```



        print1(map[root]);
        print2(map[root]);
    }
    
    public static Node getNode(int i) {
        if (i < 1) return null;
        return new Node(i);
    }
    
    public static void print1(Node head)
    {
        if (head == null)
            return;
    
        int height = getHeight(head);
        Node[][] map = new Node[height][2];
        setMap(map, head, 0);
        for (int i = 0; i < height; i ++) {
            System.out.print(map[i][0].val + " ");
        }
        printLeafNode(map, head, 0);
    
        for (int i = height - 1; i >= 0; i--) {
            if (map[i][0] != map[i][1]) {
                System.out.print(map[i][1].val + " ");
            }
        }
        System.out.println();
    }
    
    public static void printLeafNode(Node[][] map, Node head, int k)
    {
        if (head == null)
            return;
    
        if (head != map[k][0]
                && head != map[k][1]
                && head.left == null
                && head.right == null) {
            System.out.print(head.val + " ");
        }
        else
        {
            printLeafNode(map, head.left, k + 1);
            printLeafNode(map, head.right, k + 1);
        }
    }
    
    public static void setMap(Node[][] map, Node head, int k)
    {
        if(head == null)    return ;
        map[k][0] = map[k][0]==null? head: map[k][0];
        map[k][1] = head;
        setMap(map, head.left, k+1);
        setMap(map, head.right, k+1);
    }
    
    public static int getHeight(Node head)
    {
        if (head == null) return 0;
        int l = getHeight(head.left);
        int r = getHeight(head.right);
        return 1 + Math.max(l,r);
    }
    
    private static void print2(Node node)
    {
        if (node == null) return;
    
        System.out.print(node.val + " ");
        if (node.left != null && node.right != null) {
            printLeft(node.left, true);
            printRight(node.right, true);
        }
        else {
            print2(node.left == null ? node.right : node.left);
        }
    
        System.out.println();
    }
    
    private static void printRight(Node node, boolean print)
    {
        if (node == null)
            return;
    
        printRight(node.left, print && node.right == null);
        printRight(node.right, print);
        if (print || (node.left == null && node.right == null))
        {
            System.out.print(node.val + " ");
        }
    }
    
    private static void printLeft(Node node, boolean print)
    {
        if (node == null)
            return;
    
        if (print || (node.left == null && node.right == null))
        {
            System.out.print(node.val + " ");
        }
    
        printLeft(node.left, print);
        printLeft(node.right, print && node.left == null);
    }
    }




#### 面试指南 165 

```
import java.util.*;

class Node {
    int val;
     Node left;
     Node right;
    public Node(int val) {
        this.val = val;
    }
}

public class Main
{
    public static  Node getNode(int i) {
        if (i < 1) return null;
        return new  Node(i);
    }
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        Node[] map = new  Node[len + 1];
        int root = in.nextInt();
        map[root] = getNode(root);
        for (int i = 1; i <= len; i++) {
            int parent = in.nextInt();
            int l = in.nextInt();
            int r = in.nextInt();
            int value = in.nextInt();
            map[parent].val = value;
            map[l] = getNode(l);
            map[r] = getNode(r);
            map[parent].left = map[l];
            map[parent].right = map[r];
        }
        int sum = in.nextInt();
        Map<Integer,Integer> map1 = new HashMap<>();
        map1.put(0,0);
        int ret = Preorder(map[root] ,0,sum ,0,1,map1 );
        System.out.println(ret); ;
    }
```



    public static int Preorder(Node head, int maxLen,int sum, int preSum, int level, Map<Integer,Integer> map ){
        if(head == null) return maxLen;
        int curSum = preSum+ head.val;
        System.out.println(curSum);
        if (!map.containsKey(curSum))
            map.put(curSum, level);
        if(map.containsKey(curSum - sum)) {
            maxLen = Math.max(maxLen,  level -map.get(curSum - sum) );
        }
    
        int maxLen1 = Preorder(head.left, maxLen,sum, curSum , level+1 ,map);
        int maxLen2 = Preorder(head.right, maxLen,sum, curSum , level+1 ,map);
        maxLen = Math.max(maxLen, Math.max(maxLen2, maxLen1));
        if(map.containsKey(curSum) && map.get(curSum) == level)
            map.remove(curSum);
        return maxLen;
    }

}

#### 面试指南 166

```
import java.util.Scanner;

import java.util.Scanner;

class Node {
    int val;
     Node left;
     Node right;
    public Node(int val) {
        this.val = val;
    }
}
class ReturnType {
    Node node;
    int maxBSTsize;
    int max;
    int min;
    public ReturnType(Node node, int maxBSTsize, int min, int max) {
        this.node = node;
        this.maxBSTsize = maxBSTsize;
        this.max = max;
        this.min = min;
    }
```



    @Override
    public String toString() {
        return "ReturnType{" +
                "node=" + node.val +
                ", BSTsize=" + maxBSTsize +
                ", max=" + max +
                ", min=" + min +
                '}';
    }
}

```
public class Main
{
    public static  Node getNode(int i) {
        if (i < 1) return null;
        return new  Node(i);
    }
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        Node[] map = new  Node[len + 1];
        int root = in.nextInt();
        map[root] = getNode(root);
        for (int i = 1; i <= len; i++) {
            int parent = in.nextInt();
            int l = in.nextInt();
            int r = in.nextInt();
            map[l] = getNode(l);
            map[r] = getNode(r);
            map[parent].left = map[l];
            map[parent].right = map[r];
        }
        ReturnType ret = afterOrder(map[root]);
        System.out.println(ret.maxBSTsize); ;
    }
```



    public static ReturnType afterOrder(Node head){
        if(head == null) return new ReturnType(head,0 ,Integer.MAX_VALUE, Integer.MIN_VALUE);;
    
        ReturnType ldata = afterOrder(head.left);
        ReturnType rdata  = afterOrder(head.right);
    
        int min = Math.min(head.val, Math.min(ldata.min, rdata.min));
        int max = Math.max(head.val, Math.max(ldata.max, rdata.max));
    
        int maxBSTsize = Math.max(ldata.maxBSTsize,rdata.maxBSTsize );
        Node node = ldata.maxBSTsize >= rdata.maxBSTsize ? ldata .node: rdata.node;
    
        if(ldata.max < head.val && ldata.node == head.left && head.val < rdata.min  && rdata.node == head.right){
            node = head;
            maxBSTsize = ldata.maxBSTsize+ 1+ rdata.maxBSTsize;
        }
        return new ReturnType(node, maxBSTsize, min, max);
    }

```
}
```

#### 面试指南167

方法1：


    public static int bstTopoSize1(Node head) {
            if(head == null )  return 0;
            int max = maxTopo(head, head);
            max = Math.max( bstTopoSize1(head.left),max);
            max = Math.max( bstTopoSize1(head.right),max);
            return max;
    }
    
    // 计算最大拓扑结构 （此函数中的h始终指向整棵树的head，只用于传递给后续调用函数）
    public static int maxTopo(Node h, Node n) {
        if(h!=null &&n != null && isBSTNode(h,n ,n.value)){
             return maxTopo(h,n.left) + maxTopo(h,n.right)  +1;
        }
        return 0;
    }
    // 在以h为头结点的树中，二叉搜索节点n。以判断节点n是否为BST节点。
    public static boolean isBSTNode(Node h, Node n, int value) {
        if(h==null) return false;
        if(h==n) return true;
        return isBSTNode(h.value< value? h.right: h.left, n  ,value);
    }


方法 2 有一点难

    public static int bstTopoSize2(Node head) {
            HashMap<Node , Record> map = new HashMap<>();
            return postOrder(head, map);
        }
    
    public static int postOrder(Node head , HashMap<Node , Record> map) {
        if(head== null) return  0;
        int lbstSize = postOrder(head.left, map);
        int rbstSize = postOrder(head.right, map);
    
        modifyMap(head.left, head.value, map,true);
        modifyMap(head.right,  head.value,map,false);
        Record rel = map.get(head.left);
        Record rer = map.get(head.right);
    
        int lst = rel==null?0 : rel.left+ rel.right + 1;
        int rst = rer==null?0 : rer.left+ rer.right + 1;
        map.put( head,new Record(lst,rst) );
        return Math.max(lst +rst +1, Math.max(lbstSize, rbstSize));
    }
    public static int modifyMap(Node n , int v,HashMap<Node , Record> map , boolean s){
        if(n == null || !map.containsKey(n)){
            return 0;
        }
        Record record =  map.get(n);
        if( s && n.value> v || !s && n.value<v){
            map.remove(n);
            return record.left + record.right +1;
        }else {
            int minus =  modifyMap( s ? n.right: n.left, v ,map,s );
            if(s) record.left -=minus;
            else record.right -=minus;
    
            map.put(n, record);
            return  minus;
        }
    
    }


#### 面值指南 169 



    public static TreeNode[] getTwoError(TreeNode node) {
    	TreeNode[] ans = new TreeNode[2];
        if(node == null) return ans;
    
        Stack<TreeNode> stack = new Stack<>();
        TreeNode head =node;
        TreeNode pre = null;
        while(!stack.isEmpty() || head!= null){
            if(head!=null){
                stack.push(head);
                head=head.left;
            }else {
                head= stack.pop();
    
                if(pre != null && head.val < pre.val){
                    ans[0] = ans[0] == null ? pre : ans[0];
                    ans[1] = head;
                }
                pre = head;
                head=head.right;
            }
        }
        return ans;
    }
#### 面试指南170



    public static boolean isContain(TreeNode t1, TreeNode t2) {
           if(t2 ==null) return true;
           if(t1 == null) return false;
           if(t1.val == t2.val){
               if(check(t1, t2)){
                   return true;
               }
           }
           return isContain(t1.left ,t2) || isContain(t1.right, t2);
        }
        public static boolean check(TreeNode t1,TreeNode t2){
            if(t2 == null) return true;
            if(t1 == null || t1.val != t2.val) return false;
    
    	return check(t1.left, t2.left) && check(t1.right, t2.right);
    }
#### 面试指南 173  根据后续数组判断是否是搜索二叉树

```
class Solution {
    public boolean isPost(int[] postorder, int st, int end) {
        if(st == end)  return true;
        int less =-1 ,more = end;
        for (int i = st; i < end ; i++) {
            if(postorder[i] < postorder[end]) less = i;
            else more = more==end? i: more;
        }
        if(less==-1 || more== end )
            return isPost(postorder,st, end-1);
        if(less != more-1)
            return false;
        return isPost(postorder,st,less) && isPost(postorder,more, end-1);
    }
    public boolean verifyPostorder(int[] postorder) {
        if(postorder.length<=0 || postorder== null) return true;
        return isPost(postorder,0, postorder.length-1);
    }
}


```

#### 面试指南 174  判断是否是搜索二叉树 和完全二叉树

```
public static boolean isCBT(Node head){
        if(head == null) return true;
        Queue<Node> queue = new LinkedList<>();
        queue.add(head);
        boolean s = false;
        while (!queue.isEmpty()){
            Node node = queue.poll();
            if(s && (node.right!=null || node.left != null)
                    || node.left ==null && node.right!=null){
                return false;
            }
            if(node.left!= null)
                queue.add(node.left);
            if(node.right!= null)
                queue.add(node.right);
            else
                s =true;
        }
        return true;
    }
```



```
public static boolean isBST2(Node head){
        if(head == null) return true;
        Stack<Node> stack = new Stack<>();
        Node node = head;
        Node pre =null;
        while (!stack.isEmpty() || node !=null){
            if(node!= null){
                stack.push(node);
                node=   node.left;
            }else{
                node = stack.pop();
                if( pre != null && pre.value > node.value)
                        return false;
                pre = node;
                node= node.right;
            }
        }
        return true;
    }

```





#### 面试指南 175 二叉树中序遍历的下一个节点





    public static Node getLeftMost(Node node){
           if(node == null) return null;
           while(node.left!= null)
               node = node.left;
           return node;
        }
    
    public static Node getNextNode(Node node){
        if(node == null)
            return node;
        if(node.right != null){
            return getLeftMost(node.right);
        }else{
            Node parent = node.parent;
            while(parent != null && parent.left != node){
                node = parent;
                parent = node.parent;
            }
            return parent;
        }
    
    }


#### 面试指南 176 二叉树的最近公共祖先

```
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q)
            return root;
        TreeNode left =  lowestCommonAncestor(root.left, p, q);
        TreeNode right =  lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null)
            return root;
        return left!=null ? left: right;
    }
```

还有方法2  

```

```



    public static void setMap( Node head,  HashMap<Node, Node> map){
            if(head== null)     return;
            if(head.left!=null) map.put(head.left, head);
            if(head.right!=null) map.put(head.right, head);
            setMap(head.left, map);
            setMap( head.right , map);
    }
    public static int solution1(Node[] nodes,int root,  Node node1, Node node2) {
        HashMap<Node, Node> map = new HashMap<>();
        if(node1 ==null && node2 == null) return -1;
        else if(node1 == node2 ) return node1.val;
        if (node1 == null) return node2.val;
        else if(node2 == null)  return node1.val;
    
        map.put(nodes[root] , null);
        setMap(nodes[root], map);
    
        HashSet<Node > set = new HashSet<>();
    
        while(map.containsKey(node1)){
            set.add(node1);
            node1 = map.get(node1);
        }
        while(!set.contains(node2)){
            node2 = map.get(node2);
        }
        return node2.val;
    }
方法3   在直接建树的时候把每个节点的父亲节点建出来







    public static class Node{
            int val;
            Node left;
            Node right;
            Node parent;
            int level;
            public Node(int val) {
                this.val = val;
                left = null;
                right = null;
                parent = null;
                level = -1;
            }
    }
    public static int solution(Node[] nodes, Node node1, Node node2) {
        while(node1 != node2) {
            if(node1.level < node2.level)
                node2 = node2.parent;
            else if(node1.level > node2.level)
                node1 = node1.parent;
            else {
                node1 = node1.parent;
                node2 = node2.parent;
            }
        }
        return node1.val;
    }
    
    public static void getlevel(Node[] nodes) {
        for(int i = 1; i < nodes.length; i++) {
            if(nodes[i].level != -1)
                continue;
            helper(nodes[i]);
        }
    }
    
    public static int helper(Node node) {
        if(node.parent == null)
            return 0;
        if(node.level != -1)
            return node.level;
    
        node.level = 1 + helper(node.parent);
    
        return node.level;
    }
#### 面试指南 179 二叉树的最大距离



    public static class ReturnType {
            int height;
            int maxNodeSize;
    
            public ReturnType(int height, int maxNodeSize) {
                this.height = height;
                this.maxNodeSize = maxNodeSize;
            }
        }
    
        public static ReturnType process(TreeNode node) {
            if(node == null)    return new ReturnType(0,0);
            ReturnType l = process(node.left);
            ReturnType r = process(node.right);
    
            int size = Math.max(l.maxNodeSize, Math.max(r.maxNodeSize, l.height + r.height +1));
            int height = Math.max(l.height , r.height)+1;
            return  new ReturnType(height, size);
            //int height = Math.max()
        }





#### 面试指南 187  派对的最大快乐值




    static class ReturnData{
            int noX;
            int yesX;
            public ReturnData(int yesX ,int noX) {
            this.noX = noX;
            this.yesX = yesX;
        }
    }
    
    public static ReturnData process(Employee X){
        if(X ==null) return new ReturnData(0,0);
    
        int hasCur = X.happy, noCur = 0;
        if(X.subordinates.isEmpty()) return new ReturnData(hasCur,noCur);
    
    	for(Employee e :X.subordinates){
            ReturnData re =  process(e);
            noCur += Math.max( re.yesX, re.noX);
            hasCur += re.noX;
        }
        return new ReturnData( hasCur ,noCur );
    }


#### 面试指南 180 通过先序数组和中序生成后续数组

```
public  static int process(int []pre, int pi, int pj,int []in, int ni , int nj , int []after, int si,HashMap<Integer, Integer> map ){
        if(pi>pj)   return si;
        after[si--] = pre[pi];
        int i = map.get(pre[pi]);
        si =  process(pre, pj - (nj - i )+1, pj, in, i+1, nj , after, si, map);
        return process(pre, pi+1, pj - (nj - i ), in, ni, i-1 , after, si, map);
    }
```

#### 面试指南 181 二叉树的所有节点 会大数溢出  正确解法是 卡特兰数求h(N)， 求解逆元

```
import java.util.*;

public class Main{
     public static long numTrees(int n){
        if(n<2){
            return 1;
        }
        long[] num = new long[n+1];
        num[0] = 1;
        for(int i=1; i<n+1; i++){
            for(int j=1; j<i+1; j++){
                num[i] += (num[j-1]%(1e9+7)*num[i-j]%(1e9+7))%(1e9+7);
            }
        }
        long ans = (long) (num[n]%(1e9+7));
        System.out.println(ans );
        return num[n];
        }

public static void main(String[] args) {
    
    Scanner in = new Scanner(System.in);
    int len = in.nextInt();
    int modNum = (int)1e9 + 7;
    int []num = new int[len+1];
    num[0] = 1;

//         for(int i = 1; i<len+1;i++){
//             for(int j=1 ;j< i+1;j++){
//                 num[i] += ((num[i-j] % modNum) * (num[j-1] % modNum)) % modNum;
//             }
//              num[i] =  num[i] % modNum;
//         }
        numTrees(len) 

​    return;
}

}
```



#### 面试指南 计算完全二叉树的节点个数



    public int countNodes(TreeNode root) {
            if(root == null) return 0;
            int l=0;
            TreeNode node = root;
            while(node.left!=null){
                l++;
                node = node.left;
            }
            int r =0;
            node = root;
            while(node.right!=null){
                r++;
                node = node.right;
            }
            //System.out.println( root == null ? "null" : root.val + " l = "+l +", r = "+r );
    ·	if(l == r && l == 0) return 1;
        if(l == r) return (int)Math.pow(2,l+1)-1;
    
        int lcnt = countNodes(root.left);
        int rcnt = countNodes(root.right);
        //System.out.println( root == null ? "null" : root.val + " lcnt = "+lcnt +", rcnt = "+rcnt );
    
        return lcnt + rcnt + 1;
    }



#### 面试指南 183 斐波那契数列的最优解



    package mianshizhinan183;
    
    import java.io.BufferedReader;
    import java.io.IOException;
    import java.io.InputStreamReader;
    import java.util.Arrays;
    import java.util.Scanner;
    
    public class Main {
        static long mod = 1000000007;
        public static long [][] multi_matrix(long [][]a, long [][]b){
            if(a == null || b == null ) return null;
    	int row = a.length;
        int col = b[0].length;
        long [][]ans = new long[row][col];
    
        for (int i = 0; i <row ; i++) {
            for (int j = 0; j < col; j++) {
                for (int k = 0; k < a[0].length; k++) {
                    ans [i][j]+= a[i][k] * b[k][j];
                    ans [i][j]%= mod;
                }
            }
        }
        return ans;
    }
    
        public static long [][]matrix_pow(long [][]a, long n){
            int row = a.length;
            int col = a[0].length;
    
            long [][]ans = new long[row][col];
            long [][]temp = new long[row][col];
    
            for (int i = 0; i < row; i++) {
                ans[i][i] =1;
            }
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    temp[i][j] = a[i][j];
                }
            }
    
            while (n>0){
                if( (n & 0x01) == 1){
                    ans = multi_matrix(ans,temp);
                }
                n = n >> 1;
                temp = multi_matrix(temp,temp);
            }
            return ans;
        }
        public static void printMatrix(int [][]m){
            int row = m.length;
            int col = m[0].length;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    System.out.print(" "+ m[i][j] );
                }
                System.out.println();
            }
        }
        public static long fabolaqie(long n){
            long [][]mtx ={{1,1},{1,0}};
            long [][]ans = matrix_pow(mtx, n-2);
    
            long ret  = ans[0][0] + ans[0][1];
             return ret;
        }
        public static void main(String[] args) throws Exception{
            Scanner in = new Scanner(System.in);
            long cnt = in.nextLong();
            int ans = 1;
            if(cnt == 1 || cnt == 2)
                System.out.println(ans);
            else
                ans = (int)( fabolaqie(cnt) % (1e9 + 7) );
            System.out.println(ans );
        }
    }
#### 面试指南12 换零钱得最小张数

```
public static int process(int []arr, int num){
        int len = arr.length ;
        int [][]dp = new int[num+1][len+1];
        int row = dp.length;
        int col = dp[0].length;
        
        for (int i = 0; i < len + 1 ; i++) {
            dp[0][i] = 0;
        }
        for (int i = 1; i < row; i++) {
            dp[i][len] = -1;
        }

        for (int i = 1; i <=num; i++) {
            for (int j = len-1; j >=0 ; j--) {
                dp[i][j] = Integer.MAX_VALUE;

                for (int k = 0; k* arr[j] <= i ; k++) {
                    if(dp[i- k*arr[j] ][j+1] == -1)
                        continue;
                    else
                        dp[i][j] = Math.min(dp[i][j], dp[i- k*arr[j] ][j+1] + k);
                }

                if(dp[i][j] == Integer.MAX_VALUE)
                    dp[i][j] = -1;
            }
        }

        return dp[num][0];
    }
```



#### 面试指南17题 机器人到达指定路径得方法数



    public static int process(int n, int target, int cur,int rest){
            int [][]dp = new int[n+1][rest+1];
    	dp[target][0]=1;
    
        for (int j = 1; j <=rest; j++) {
            for (int i = 1; i <=n; i++) {
                if(i == 1)
                    dp[i][j] = (dp[i+1][j-1] % NUM);
                else if(i == n)
                    dp[i][j] = (dp[i-1][j-1])% NUM;
                else
                    dp[i][j] = (dp[i-1][j-1])% NUM + (dp[i+1][j-1])% NUM;
            }
        }
    
        return (dp[cur][rest]% NUM);
    }



####  面试指南19题 换钱得方法数

    public static int process1(int []arr, int n, int num , int cur){
            int [][]dp = new int[num+1][n+1];
    	dp[0][n] = 1;
        for (int j = n-1; j>=0 ; j--) {
            for (int i = 0; i <=num; i++) {
                for (int k = 0; k* arr[j] <= i ; k++) {
                    dp[i][j] += dp[i- k*arr[j]][j+1];
                    dp[i][j] %= MOD;
                }
            }
        }
    
        return dp[num][0]% MOD;
    }
​                                                                                                                                                                                                              

#### 面试指南 95 异位词

```
public static boolean isDeformation(String s1, String s2) {
        if (s1 == null || s2 == null || s1.length() != s2.length()) {
            return false;
        }
        char [] arr1 = s1.toCharArray();
        char[]arr2 = s2.toCharArray();
        int []map = new int[256];

​    	for(char c: arr1){
​        map[c]++;
​    }
​    for(char c: arr2){
​        if(map[c]-- == 0)
​            return false;
​    }
​    return true;
}
```

#### 面试指南 96 旋转词

    public static boolean isRoate(String str1, String str2) {
        if (str1 == null || str2 == null
                || str1.length() != str2.length()) {
            return false;
        }
        String ss =  str2 + str2;
    
        if (ss.contains(str1)) {
            return true;
        }
        return false;
    }




#### 面试指南 97 把字符串转化为值 

```
public static boolean isValid(char[] ch){
       if(ch == null || ch.length<=0) return false;
       int len = ch.length ,i=0;

   	if(ch[i] == '-' || ch[i]>='1' && ch[i] <= '9') i++;
  	 else return false;
   	if(i == len ) return false;
   	while (i<len){
  	     if(ch[i]>='0' && ch[i] <= '9' ) i++;
 	      else return false;
  	 }
 	  return true;
}

public static int convert(String str){
    if(str == null || str == ""){
        return 0;
    }
    char[] ch = str.toCharArray();
    if(!isValid(ch)){
        return 0;
    }
    boolean neg = false;
    neg = ch[0] == '-' ? true : false;

​    int len = ch.length;
​    int i = neg == true ? 1: 0;
​    if(len - i > 10) return 0;
​    int ans = 0 , cnt =0;

​    while (i<len){
​        ans = ans * 10 + ch[i] - '0';
​        cnt++;
​        if(cnt == 9 && len -i == 2){
​            int thNeg = Integer.MIN_VALUE /10;
​            int negMod = Integer.MIN_VALUE % 10;
​            //System.out.println("negMod " + negMod);
​            int th = Integer.MAX_VALUE /10;
​            int Mod = Integer.MAX_VALUE % 10;
​            //System.out.println("Mod " + Mod);
​            if(neg && (-ans < thNeg || (-ans == thNeg &&  -(ch[i+1] -'0')< negMod) ) ) return 0;
​            if(!neg &&(ans == th && ch[i+1]-'0' > Mod  || ans > th ) )
​                return 0;
​        }

​        i++;
​    }

​    return neg ? -ans:ans;
}
```





#### 面试指南啊 98 字符串字符统计

    public static String process(String str){
    	if(str.length() == 0 || str == null){
            return "";
        }
        StringBuilder res = new StringBuilder();
        char[] chs = str.toCharArray();
        int i=0 , j =0;
        while (i<  chs.length){
            j=i;
            while(j<chs.length && chs[j] == chs[i]){ j++; }
            res.append(chs[i]).append('_').append(j-i);
            if(j != chs.length) res.append('_');
            i = j;
        }
        return res.toString();
    }


#### 面试指南103 堆排序



    public static void swap(char [] arr,  int id1, int id2){
            char c = arr[id1];
            arr[id1] = arr[id2];
            arr[id2] = c;
        }
        public  static void heapInsert(char [] arr,  int i){
            int parent = 0;
            while (i!=0){
                parent = (i-1)/2;
                if(arr[parent] < arr[i]){
                    swap(arr,parent, i);
                    i = parent;
                }else
                    break;
            }
        }
        public static void heapify(char [] arr, int i, int size){
            int left = i*2 +1;
            int right = i*2 +2;
            int largest = i;
    	while(left < size){
            if(arr[left] > arr[right]){
                largest = left;
            }
            if(right <size  && arr[right] > arr[largest]){
                largest = right;
            }
            if(largest != i){
                swap(arr,largest,i );
            }else
                break;
            i = largest;
            left = i* 2+1;
            right = i*2+2;
        }
    }
    public static void heapSort(char[] arr){
        for (int i = 0; i < arr.length; i++) {
            heapInsert(arr, i);
        }
        for (int i = arr.length-1 ;i >0  ; i--) {
            swap(arr, 0, i);
            heapify(arr, 0,i);
        }
    }
    
    public static Boolean isUnique(String[] arr){
        if(arr==null||arr.length==0){
            return true;
        }
        String str = new String();
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length(); j++) {
                if(arr[i].charAt(j) != ' ') str = str+ arr[i].charAt(j);
            }
        }
        char []arr1 = str.toCharArray();
        heapSort(arr1);
        for (int i = 1; i < arr1.length; i++) {
            if(arr1[i] == arr1[i-1])
                return false;
        }
    
        return true;
    }
#### 面试指南 99 有序字符串中查找最先出现的字符串 

```
public static int process(String[] arr, String str){
        if(arr == null || str == null || arr.length == 0){
            return -1;
        }
        int left =0, right = arr.length-1;
        int mid = 0;
        int ans =-1, i =  0;
        while(left <= right){
            mid = (right +left)/2;
            if(arr[mid] != null && arr[mid].equals(str)){
                ans = mid;
                right= mid -1;
            }else if(arr[mid] != null){
                if(arr[mid].compareTo(str) < 0){
                    left = mid +1;
                }else {
                    right = mid -1;
                }
            }else {
                i=mid;
                while(arr[i] == null && --i>=left);
                if(i < left || arr[i].compareTo(str) <0){
                    left = mid+1;
                }else {
                    ans = arr[i].equals(str)? i: ans;
                    right = i-1;
                }
        }
    }
    return ans;
}
```



#### 面试指南 113  双指针

```
public static String retota(char[] chas){
        if(chas==null || chas.length <=0)
            return "";
        int i=0,j=0;
        while(i< chas.length ){
            while(i< chas.length && chas[i]==' '){
                i++;
            }

​        j=i;
​        while (j<chas.length && chas[j] != ' '){
​            j++;
​        }
​        reserve(chas, i,j-1);
​        i=j;
​    }
​    return new String(chas);
}

public static void reserve(char[] chas, int start, int end){
    while (start<end){
        char c = chas[start];
        chas[start] = chas[end];
        chas[end] = c;
        start++;
        end--;
    }
}
```



#### 面试指南118 完美洗牌问题

```
public class Main{

    public static int modifyIndex(int i, int len){
        return (2 * i) % (len + 1);
    }
    
    public static void reverse(int[] arr, int L, int R){
        while(L < R){
            int tmp = arr[L];
            arr[L++] = arr[R];
            arr[R--] = tmp;
        }
    }
    
    public static void rotate(int[] arr, int L, int M, int R){
        reverse(arr, L, M);
        reverse(arr, M + 1, R);
        reverse(arr, L, R);
    }
    
    // 从start位置开始，往右len的长度这一段做下标连续推
    // 出发位置依次为1，3，9....
    public static void cycles(int[] arr, int start, int len, int k){
    
        for (int i = 0, base =1; i < k; i++, base *=3) {
    
            int preValue = arr[base];
            int cur = modifyIndex(base, len);//base的下一个位置
    
            while (cur != base){
                int temp = arr[cur + start -1];
                arr[cur + start -1] = preValue;
                preValue = temp;
                cur = modifyIndex(cur,len);
            }
            arr[cur + start -1] = preValue;
        }
    }
    
    public static void shuffle(int[] arr, int L, int R){
        int len = R - L +1 ;
        int base = 3;
    
        while (len > 0){
    
            int k = 1;
            int cnt = 1;
            while (k < len -1){
                k*=base;
                cnt++;
            }
            int mid = (L+ R)/2;
            int half = (k-1)/2;
            rotate(  arr ,L + half, mid , mid + half );
            cycles(arr,L, k-1, cnt );
            L = L +k-1;
            len = R - L +1;
        }
    
    }
    
    public static void shuffle(int[] arr){
        if(arr !=null && arr.length !=0 && ((arr.length & 1) == 0) ){
            shuffle(arr, 0, arr.length -1);
        }
    }
    
    public static void main(String[] args) throws Exception{
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(in.readLine());
        String[] ops = in.readLine().split(" ");
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < n/2; i++){
            if(i>0){
                sb.append(" ");
            }
            sb.append(ops[n/2 + i]);
            sb.append(" ");
            sb.append(ops[i]);
        }
        System.out.println(sb.toString());
    }

}
```



#### 面试指南 117 构造最小的字典序

```
public class Main{
    public static String removeDuplicateLetters(String s) {
        if(s == null ||s.length() <=0)  return null;
        int []map = new int[26];
        char []arr = s.toCharArray();
        for (int i = 0; i < s.length(); i++) {
            map [ arr[i] - 'a'] ++;
        }
        int L =0, R = 0, index =0;
        char []res = new char[26];
        while(R != arr .length){
            int id = arr[R] - 'a';
            if(map[id] == -1 || --map[id] >0){
                R++;
            }else{
                int pick = -1;
                for (int i = L; i <= R; i++) {
                    if(map[id] != -1 && ( pick ==-1 || arr[i] < arr[pick] ) ){
                        pick = i;
                    }
                }
                res[index++] = arr[pick];
                for (int i = pick+1; i <=R ; i++) {
                    if(map[ arr[i] - 'a'] != -1 ){
                        map[arr[i] - 'a']++;
                    }
                }
                map[arr[pick] - 'a'] = -1;
                L = pick+1;
                R = L;
            }

        }
        return new String(res);
    }
    public static void main(String[] args) {
        System.out.println(removeDuplicateLetters("acbc"));
    }

}
```

#### 面试指南122 字符串中的最小距离

```
public static int solution(String[] nums, String str1, String str2) {
        int res = Integer.MAX_VALUE;
        int pos1 = -1;
        int pos2 = -1;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i].equals(str1)){
                res = pos2 == -1 ? res : Math.min(res, i - pos2);
                pos1 = i;
            }else if(nums[i].equals(str2)){
                res = pos1 == -1 ? res : Math.min(res,  i - pos1);
                pos2 = i;
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }
```

#### 面试指南 123 寻找字符串之间的所有最短路径

这道题代码太多了，暂时没来得及动手敲先下一个



#### 面试指南啊 124 添加最少字符使之成为回文 ，返回最终回文字符串

```
public static int[][] getDp(String s){
        char[] str = s.toCharArray();
        int n = str.length;
        int[][] dp = new int[n][n];

​    for (int j = 1; j < n ; j++) {
​        for (int i = j-1;i>=0 ; i--) {
​            if(str[i] == str[j])
​                dp[i][j] = dp[i+1][j-1];
​            else
​                dp[i][j] = Math.min(dp[i+1][j], dp[i][j-1])+1;
​        }
​    }
​    return dp;
}
public static String process(String str) {
​    if (str == null || str.length() < 2) {
​        return str;
​    }
​    char[] chas = str.toCharArray();
​    int[][] dp = getDp(str);
​    char[] res = new char[chas.length + dp[0][chas.length - 1]];
​    int i = 0;
​    int j = chas.length - 1;
​    int resl = 0;
​    int resr = res.length - 1;
​    while (i <= j) {
​        if(chas[i] == chas[j]){
​            res[resl++] = chas[i++];
​            res[resr--] = chas[j--];
​        }else if(dp[i+1][j] < dp[i][j-1]){
​            res[resr--] = chas[i];
​            res[resl++] =chas[i++];

​        }else{
​            res[resr--] = chas[j];
​            res[resl++] =chas[j--];
​        }
​    }
​    return String.valueOf(res);
}
```

#### 面试指南啊 124 增加一个最大回文自诩了，返回最终回文字符串

```
public static String solution(String str1, String str2) {
        char[] chs1 = str1.toCharArray();
        char[] chs2 = str2.toCharArray();

​    int len = chs1.length * 2 - chs2.length;
​    char[] res = new char[len];

​    int s2left = 0, s2right = str2.length()-1;
​    int s1left = 0, s1right = str1.length()-1;

​    int templeft =s1left,tempright = s1right;

​    int left =0, right = res.length -1;
​    while (s2left <= s2right){
​        templeft = s1left;
​        tempright = s1right;

​        while ( chs1[s1left]!= chs2[s2left]) s1left++;
​        while ( chs1[s1right]!= chs2[s2right]) s1right--;

​        for (int i = templeft; i <= s1left-1; i++) {
​            res[left++] = chs1[i] ;
​            res[right--] = chs1[i] ;
​        }
​        for (int i = tempright ; i >= s1right+1; i--) {
​            res[left++] = chs1[i] ;
​            res[right--] = chs1[i] ;
​        }

​        res[left++] = chs1[s1left++] ;
​        res[right--] = chs1[s1right--] ;

​        s2left++;
​        s2right--;
​    }

​    return String.valueOf(res);
}
```



#### 面试指南126 括号的有效性

```
public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader reader=new BufferedReader(new InputStreamReader(System.in));
        String str=reader.readLine();
        System.out.print(isValid(str)?"YES":"NO");
    }
    public static boolean isValid(String str){
        char[] chars=str.toCharArray();
        int num=0;
        for (int i = 0; i < chars.length; i++) {
            if(chars[i] != ')' && chars[i] != '(')
                return false;
            if(chars[i] == ')'){
                num--;
                if(num<0)  return false;
            }else if(chars[i]=='(')
                num++;
        }
        return num==0?true:false;
    }
}
```

#### 面试指南127 括号的最大长度

注意如果() (()) 如果遍历到最后一个括号要把前面的两个 括号加上

```
private static int checkArr(char[] arr){
        if(arr==null||arr.length<2) return 0;

​    int[] dp = new int[arr.length+1];
​    int res=0;
​    dp[0] = 0;
​    for (int i = 1; i < arr.length; i++) {
​        if(arr[i] == ')'){
​            int pre = i - dp[i-1]  -1;
​            if(pre >=0 &&  arr[pre] == '(') {
​                dp[i] = dp[i - 1] + 2 + (pre > 0 ? dp[pre - 1] : 0);
​            }
​            res = Math.max(res, dp[i]);
​        }
​    }
​    return res;
}
```

#### 面试指南130 字符串构成的最小字典序



```
public static void main(String args[]) throws IOException {
        int N = Integer.parseInt(in.readLine()) ;
        List<String> tmp = new ArrayList<>();
        for(int i = 0; i < N; i ++) {
            String t = in.readLine();
            tmp.add(t);
        }
        Collections.sort(tmp, new Comparator<String>() {
            public int compare(String a, String b) {
                return (a+b).compareTo(b + a);
            }
        });
        StringBuffer sb = new StringBuffer();
        for(int i = 0; i < N; i ++) {
            sb.append(tmp.get(i));
        }
        System.out.println(sb.toString());
    }
```

#### 面试指南 131 滑动窗口 

```
public static int process(int []arr ){
        HashMap<Integer, Integer> map = new HashMap<>();
        int right=0,left=0;
        int ans = 0;
        while(right<arr.length){
            map.put(arr[right], map.getOrDefault(arr[right],0)+1);
            while(map.getOrDefault(arr[right],0) > 1){
                ans = Math.max(ans, right- left);
                map.put(arr[left], map.getOrDefault(arr[left],0)-1);
                left++;
            }
            right++;
        }
        ans = Math.max(ans, right- left);
        return ans;
    }
```



#### 面试指南132 找到新类型的字符

```
public static String process(String str, int n, int k){
        if(n <=0 || k < 0 || k > n || str == null){
            return "";
        }
        char[] chas = str.toCharArray();
        int nNum = 0;
        int i = k-1;
        while (i>=0 && Character.isUpperCase(chas[i])){
            i--;
        }
        nNum = k-1 -i;
        if((nNum & 1) == 1){ //nNum是奇数
            return str.substring(k-1, k+1);
        }
        else if(Character.isUpperCase(chas[k])  && (nNum & 1) == 0 ){//若是偶数，且当前为大写
            return str.substring(k, k+2);
        }
        else return String.valueOf(chas[k]);
    }
```

#### 面试指南 133 旋转字符串

```
public static boolean solution(String str1, String str2) {
        if(str1.length() != str2.length())
            return false;
        if(str1.length() == 1)
            return str1.equals(str2);

​    char[] chs1 = str1.toCharArray();
​    char[] chs2 = str2.toCharArray();

​    int[] cnt = new int[128];
​    for(char ch : chs1)
​        cnt[ch]++;
​    for(char ch : chs2){
​        if(--cnt[ch] < 0)
​            return false;
​    }

​    for(int end = 0; end < chs1.length - 1; end++) {
​        if(solution(str1.substring(0, end + 1), str2.substring(0, end + 1))
​                && solution(str1.substring(end + 1), str2.substring(end + 1))
​                || solution(str1.substring(0, end + 1), str2.substring(str2.length() - 1 - end))
​                && solution(str1.substring(end + 1), str2.substring(0, str2.length() - 1 - end)) )
​            return true;
​    }
​    return false;
}
```



Leetcode 460 LFU缓存



[]()

    class LFUCache {
        Map<Integer, Node> cache;  // 存储缓存的内容
        Map<Integer, LinkedHashSet<Node>> freqMap; // 存储每个频次对应的双向链表
        int size;
        int capacity;
        int min; // 存储当前最小频次
    
    public LFUCache(int capacity) {
        cache = new HashMap<> (capacity);
        freqMap = new HashMap<>();
        this.capacity = capacity;
    }
    
    public int get(int key) {
        Node node = cache.get(key);
        if (node == null) {
            return -1;
        }
        freqInc(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        if (capacity == 0) {
            return;
        }
        Node node = cache.get(key);
        if (node != null) {
            node.value = value;
            freqInc(node);
        } else {
            if (size == capacity) {
                Node deadNode = removeNode();
                cache.remove(deadNode.key);
                size--;
            }
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            addNode(newNode);
            size++;     
        }
    }
    
    void freqInc(Node node) {
        // 从原freq对应的链表里移除, 并更新min
        int freq = node.freq;
        LinkedHashSet<Node> set = freqMap.get(freq);
        set.remove(node);
        if (freq == min && set.size() == 0) { 
            min = freq + 1;
        }
        // 加入新freq对应的链表
        node.freq++;
        LinkedHashSet<Node> newSet = freqMap.get(freq + 1);
        if (newSet == null) {
            newSet = new LinkedHashSet<>();
            freqMap.put(freq + 1, newSet);
        }
        newSet.add(node);
    }
    
    void addNode(Node node) {
        LinkedHashSet<Node> set = freqMap.get(1);
        if (set == null) {
            set = new LinkedHashSet<>();
            freqMap.put(1, set);
        } 
        set.add(node); 
        min = 1;
    }
    
    Node removeNode() {
        LinkedHashSet<Node> set = freqMap.get(min);
        Node deadNode = set.iterator().next();
        set.remove(deadNode);
        return deadNode;
    }
}

class Node {
    int key;
    int value;
    int freq = 1;

    public Node() {}
    
    public Node(int key, int value) {
        this.key = key;
        this.value = value;
    }
}

作者：sweetiee
链接：https://leetcode-cn.com/problems/lfu-cache/solution/java-13ms-shuang-100-shuang-xiang-lian-biao-duo-ji/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#### leetcode 146 LRU 缓存机制





    class LRUCache {
            static LinkedHashMap<Integer ,Integer> link;
            static int Size;
            public LRUCache(int capacity) {
                link = new LinkedHashMap<>();
                Size= capacity;
            }
    
    	public int get(int key) {
            if(link.containsKey(key)) {
                Integer integer = link.get(key);
                link.remove(key);
                link.put(key,integer);
                return integer;
            }
            return -1;
        }
    
    //linkListhashmap 默认新插入的节点为尾部插入，所以淘汰新节点的时候就把首部给删除就行了
        public void put(int key, int value) {
            if( link.containsKey(key)) {
                link.remove(key);
                link.put(key,value);
            }
            else if(link.size()<Size )
                link.put(key,value);
            else{
                Map.Entry<Integer, Integer> next = link.entrySet().iterator().next();
                link.remove(next.getKey());
                link.put(key,value);
            }
        }
    }
```
class ListNode{
    int key;
    int value;
    ListNode pre;
    ListNode next;

    public ListNode(int value, ListNode pre, ListNode next) {
        this.value = value;
        this.pre = pre;
        this.next = next;
    }

}
class Mylist{
    ListNode head = null;
    ListNode tail = null;


    public ListNode insertHead(int data){
        ListNode  node = new ListNode(data, null, null);
        if(head ==null) {
            head = node;
            tail = node;
        }else if(head!=null){
            head.pre = node;
            node.next = head;
            head  = node;
        }
        return node;
    }
    public ListNode insertHead(ListNode node){
        if(head ==null) {
            head = node;
            tail = node;
        }else if(head!=null){
            head.pre = node;
            node.next = head;
            node.pre =null;
            head  = node;
        }
        return node;
    }
    public ListNode removeNode(ListNode n1){
    
        if(head == tail && n1 == tail) {
            head = null;
            tail = null;
        }else if(head == n1){
            head = head.next;
            head.pre = null;
        }else if(n1 == tail){
            tail.pre.next = null;
            tail = tail.pre;
        }else {
            n1.pre.next = n1.next;
            n1.next.pre = n1.pre;
        }
        return n1;
    }
    public ListNode deleteTail(){
        if(tail == null )
            return null;
        if(tail == head ){
            tail = head = null;
        }else {
            tail.pre=null;
            tail = tail.pre;
        }
        return tail;
    }

}
class LRUCache {

    int size;
    Mylist list;
    HashMap<Integer , ListNode> map ;
    public LRUCache(int capacity) {
        size = capacity;
        map = new HashMap<>();
        list = new Mylist();
    }
    public int get(int key) {
        if (map.containsKey(key)) {
        //删除字符 ，提到前面来
            ListNode node = map.get(key);
            list.removeNode(node);
            list.insertHead(node);
            return node.value;
        }
        else
            return -1;
    }
    public void put(int key, int value) {
    
        if(map.containsKey(key)){
            ListNode node = map.get(key);
            node.value = value;
            //删除这个节点
            list.removeNode(node);
            list.insertHead(node);
        }
        else if(map.size() < size ) {
            ListNode node = new ListNode(value,null,null );
            node.key = key;
            list.insertHead(node);
            map.put(key, node);
    
        }else if(map.size() >= size){
            //把之前的删掉
            ListNode node = list.tail;
            list.removeNode(node);
            map.remove(node.key);
    
            ListNode node1 = new ListNode(value,null,null );
            node1.key = key;
            list.insertHead(node1);
            map.put(key, node1);
        }
    }

}
```





#### leetcode  152 数组的连续子数组的最大乘积

```
public int maxProduct(int[] nums) {
        int [] maxdp =new  int[ nums.length];
        int [] mindp =new  int[ nums.length];
        for (int i = 0; i < nums.length; i++) {
            maxdp[i] = nums[i];
            mindp[i] = nums[i];
        }
        int ans = maxdp[0];
        for (int i = 1; i < nums.length; i++) {
            maxdp[i] = Math.max( nums[i]* maxdp[i-1] , Math.max( nums[i] ,nums[i]* mindp[i-1])  );
            mindp[i] = Math.min( nums[i]* maxdp[i-1] , Math.min( nums[i] ,nums[i]* mindp[i-1])  );

            ans = Math.max(ans, maxdp[i]);
        }
        return ans ;
    }


```

#### leetcode  155 Min函数的栈





    	Stack<Integer> stack;
        Stack<Integer> minStack;
    	public MinStack() {
            stack = new Stack<>();
            minStack = new Stack<>();
    
        }
        public void push(int x) {
            stack.push(x);
            if(minStack.isEmpty()) {
                minStack.push(x);
            }else {
                minStack.push( Math.min(x, minStack.peek()));
            }
        }
    
        public void pop() {
            stack.pop();
            minStack.pop();
        }
    
        public int top() {
            return stack.peek();
        }
    
        public int getMin() {
            return minStack.peek();
        }








#### leetcode 169  水王数 boyer-moore 投票算法

​        

     public int majorityElement(int[] nums) {
            int n = nums.length;
            if (n< 3) return nums[0];
     	int c = 1;
        int m = nums[0];
        for (int i = 1; i <n ; i++) {
            if(c == 0){
                m = nums[i];
            }
            if(m == nums[i]) c++;
            else  c--;
        }
        return m;
    }
#### leetcode  198  强盗抢劫金额最大问题 



    public int rob(int[] nums) {
            if(nums == null || nums.length <= 0)
                return  0;
            int [] dp = new int[nums.length+1];
            dp[nums.length] =0;
            dp[nums.length-1] = nums[nums.length-1];
            for (int i = nums.length-2; i >=0 ; i--) {
                dp[i] = Math.max(nums[i] + dp[i+2],  dp[i+1]);
            }
            return dp[0];
    }



#### 面试指南 140   字典 前缀树



    static class TrieNode{
            int end;
            int path;
            TrieNode []map;
    
    	public TrieNode() {
            this.end = 0;
            this.path = 0;
            map = new TrieNode[26];
        }
    }
    
    public static class Trie{
        private TrieNode root;
        public Trie(){
            root = new TrieNode();
        }
        public void insert(String word){
            if(word == null )return;
            char [] arr = word.toCharArray();
            TrieNode node = root;
            node.path++;
    
            for (int i = 0; i <arr.length ; i++) {
                int index = arr[i] - 'a';
                if(node.map[index] == null){
                    node.map[index] = new TrieNode();
                }
                node = node.map[index];
                node.path++;
            }
            node.end++;
        }
    
        public void delete(String word){
            if(word == null )return;
            if(search(word)){
                char [] arr = word.toCharArray();
                TrieNode node = root;
                node.path++;
    
                for (int i = 0; i <arr.length ; i++) {
                    int index = arr[i] - 'a';
                    if(node.map[index].path -- == 1){
                        node.map[index] = null;
                        return;
                    }
                    node = node.map[index];
                }
                node.end--;
    
            }
        }
    
        public boolean search(String word){
            if(word == null )return false;
    
            char [] arr = word.toCharArray();
            TrieNode node = root;
            for (int i = 0; i <arr.length ; i++) {
                int index = arr[i] - 'a';
                if(node.map[index] == null){
                    return false;
                }
                node = node.map[index];
            }
            return node.end != 0;
        }
    
        public int prefixNumber(String pre){
            if(pre == null )return 0;
    
            char [] arr = pre.toCharArray();
            TrieNode node = root;
            for (int i = 0; i <arr.length ; i++) {
                int index = arr[i] - 'a';
                if(node.map[index] == null){
                    return 0;
                }
                node = node.map[index];
            }
            return node.path;
    
        }
    }
#### 面试指南 141  最大异或和 （前缀树）



```
package mianshizhinan141;

import java.io.*;
import java.util.*;

public class Main{
    static Node head;
    public static class Node{
        Node []next = new Node[2];
    }

    public static void add(int num) {
    
        Node node = head;
    
        for (int i = 31; i >=0 ; i--) {
            int id = (num>>i) & 1;
            if(node.next[id] == null){
                node.next[id] = new Node();
            }
            node = node.next[id];
        }
    }
    
    public static int getmax(int num) {
        Node node = head;
        int ans = 0;
        for (int i = 31; i >=0 ; i--) {
            int id = (num>>i) & 1;
            int expect = i == 31 ? id : id ^1;
            int real = node.next[expect] != null ? expect: expect^1;
    
            ans |= (id ^real)<<i;
    
            node = node.next[real];
        }
        return ans;
    }



    public static int solution(int[] nums) {
        //trie
        head = new Node();
        add(0);
    
        int res = Integer.MIN_VALUE;
        int xor = 0;
        for(int num : nums) {
            xor ^= num;
            res = Math.max(res, getmax(xor));
    
            add(xor);
        }
        return res;
    }


    public static void main(String[] args) throws Exception {
        BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(bf.readLine());
    
        //String[] strs = bf.readLine().trim().split(" ");
        int[] nums = new int[n];
        for(int i = 0; i < n; i++)
            nums[i] = Integer.parseInt(bf.readLine());
    
        System.out.println(solution(nums));
    }

}
```



第二种解法 

```
public  static int process(int[] nums){
        if(nums.length <= 0 || nums == null)
            return  0;
        int [] eor = new int[nums.length];
        eor[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            eor[i] = nums[i] ^ eor[i-1];
        }
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < eor.length; i++) {
            for (int j = 0; j <=i; j++) {
                max = Math.max(max,   j == 0 ? eor[i] : eor[i]^ eor[j-1]);
            }
        }
        return max;

    }
```



#### 面试指南 148  出现k次的数只出现一次的数

```
public class Main{
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String[] s = br.readLine().split(" ");
        int n = Integer.parseInt(s[0]);
        int k = Integer.parseInt(s[1]);
        int[] arr = new int[n];
        s = br.readLine().split(" ");
        for (int i=0; i<n; i++){
            arr[i] = Integer.parseInt(s[i]);
        }

        int num = getNum(arr,k);
        System.out.println(num);
    }
    
    public static int getNum(int[] arr, int k) {
        //记录各个位上%k之后的数
        int[] count = new int[32];
    
        for (int i=0; i<arr.length; i++) {
            //将arr中的每个数字化成k进制的数
            int []res =kSystem( arr[i], k);
            for (int j = 0; j < 32; j++) {
                count[j] = (count[j] + res[j]) % k;
            }
        }
        //将count表示的k进制数转化为10进制数，即为出现一次的数
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res += count[i] * Math.pow(k, i);
        }
        return res;
    }
    public static int[] kSystem(int num, int k){
          int [] count = new int[32];
          int i=0;
    
          while (num > 0){
              count[i++] = num % k;
              num = num / k;
          }
          return count;
    }

}


```



#### 面试指南 151 转圈打印矩阵 



```
public static void printMatrixBySpin(int[][] matrix) {
        int tR = 0;
        int tC = 0;
        int dR = matrix.length - 1;
        int dC = matrix[0].length - 1;
        while (tR <= dR && tC <= dC) {
            printEdge(matrix, tR++, tC++, dR--, dC--);
        }
    }

    public static void printEdge(int[][] matrix, int tR, int tC, int dR, int dC) {
        if (tR == dR) {
            for (int i = tC; i <= dC; i++) {
                System.out.print(matrix[tR][i] + " ");
            }
        } else if (tC == dC) {
            for (int i = tR; i <= dR; i++) {
                System.out.print(matrix[i][tC] + " ");
            }
        } else {
            int curR = tR;
            int curC = tC;
            while (curC != dC) {
                System.out.print(matrix[tR][curC++] + " ");
            }
            while (curR != dR) {
                System.out.print(matrix[curR++][dC] + " ");
            }
            while (curC != tC) {
                System.out.print(matrix[dR][curC--] + " ");
            }
            while (curR != tR) {
                System.out.print(matrix[curR--][tC] + " ");
            }
        }
    }

```



#### 面试指南  150  转圈打印矩阵



```
private static void rotate(String[][] arr){
       if(arr == null || arr.length < 2) return;
       int tR =0, tC = 0;
       int dR = arr.length -1, dC = arr[0].length-1;

       while (tR <= dR && tC <= dC){
           rotateEdge(arr, tR++, tC++, dR--, dC--);
       }
    
    }
    //temp = arr[tr][tc+i];

//        arr[tr][tc+i] = arr[dr-i][tc];
//        arr[dr-i][tc] = arr[dr][dc-i];
//        arr[dr][dc-i] = arr[tr+i][dc];
//        arr[tr+i][dc] = temp;
    private static void rotateEdge(String[][] arr,int tr,int tc,int dr,int dc){
        int t = dc-tc;
        String temp;
        for(int i=0;i<t;i++){
            temp = arr[dr-i][tc];
            arr[dr-i][tc] = arr[dr][dc-i];
            arr[dr][dc-i] = arr[tr+ i][dc];
            arr[tr+ i][dc] = arr[tr][tc+i];
            arr[tr][tc+i] = temp;
        }
    }
```



#### 面试指南 151 zigzig打印矩阵 



```
public static void printLevel(int[][] m, int tR, int tC, int dR, int dC, boolean f, StringBuilder sb){
        if(f){
            while (tR != dR +1){
                sb.append( m[tR++][tC--] + " ");
            }
        }else{
            while (dR != tR - 1){
                sb.append( m[dR--][dC++] + " ");
            }
        }
    }

    public static void printMatrixZigZag(int[][] matrix){
        if(matrix == null  ) return;
        boolean f = false;
        int tR = 0, tC = 0;
        int dR = 0, dC = 0;
    
        int endR = matrix.length -1, endC = matrix[0].length-1;
        StringBuilder sb = new StringBuilder();
        while (dC != endC +1){
            printLevel(matrix, tR , tC, dR ,dC , f , sb);
            //System.out.println( "tR , tC, dR ,dC " + tR +" " + tC+" " +dR +" "+dC);
            tR = tC == endC ? tR+1 : tR;
            tC = tC == endC ? tC : tC+1;
            
            dC = dR == endR ? dC +1: dC;
            dR = dR == endR ? dR : dR+1;
            f = !f;
    
        }
        System.out.println(sb.toString());
    }

```



#### 面试指南  153 排序的最短子数组



```
public  static  int process( int [] nums){
        if(nums == null || nums .length < 2) return 0;
        int minId = -1 , min = nums[nums.length-1];
        for (int i = nums.length  -2; i >=0 ; i--) {
            if(nums [i]> min ){
                minId = i;
            }else {
                min  = Math.min(min,  nums[i]);
            }
        }
        if(minId == -1)     return 0;
        int maxId = nums.length , max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if(nums[i] < max){
                maxId = i;
            }else{
                max  = Math.max(max,  nums[i]);
            }
        }
        return maxId - minId + 1;
    }


```



####  面试指南 154 寻找数组中大于一半次数的数

```
 private static String getNum(String[] arr){
        if(arr==null||arr.length<1) return "-1";
        if(arr.length==1) return arr[0];

        String card= arr[0];
        int times= 1;
        for(int i=1;i<arr.length;i++){
             if(times == 0){
                 card = arr[i];
             }
             times =  arr[i].equals(card) ? times+1: times-1;
        }
        times =0;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i].equals(card))
                times++;
        }
        return  (times > arr.length/2)? card: "-1";
    }


```

#### 面试指南 155 寻找 次数大于 n/k的数  只过了15个case

```
public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        String[] input = br.readLine().split(" ");
    
        int n = Integer.parseInt(input[0]);
        int k = Integer.parseInt(input[1]);
        int val = n / k + 1;
    
        int[] arr = new int[n];
        input = br.readLine().split(" ");
    
        for(int i = 0; i < n; i++){
            arr[i] = Integer.parseInt(input[i]);
        }
    
        Map<Integer, Integer> map = new HashMap<>();
        StringBuilder builder = new StringBuilder();
    
        for (int i = 0; i < arr.length; i++) {
            if(map.containsKey(arr[i])){
                map.put(arr[i], map.get(arr[i])+1);
            }else {
                if(map.size() == k-1){
                    List<Integer> list = new LinkedList<>();
                    for(Map.Entry<Integer, Integer> en :map.entrySet() ){
                         if(map.get(en.getKey())==1)  list .add (en.getKey());//直接remove会内存出错
                         else map.put(en.getKey(),map.get(en.getKey())-1);
                    }
                    for (int id : list) {
                        map.remove(id);
                    }
                }else{
                    map.put(arr[i], 1);
                }
            }
        }
    
        HashMap<Integer, Integer> real = new HashMap<>();
    
        for (int i = 0; i < arr.length; i++) {
    
            if(map.containsKey(arr[i])) {
                if(real.containsKey(arr[i])){
                    real.put(arr[i], real.get(arr[i])+1);
                }else{
                    real.put(arr[i],1);
                }
            }
        }
    
        boolean hasPrint = false;
        String ans = new String();
        for (Map.Entry<Integer, Integer> entry: map.entrySet()){
            if(entry.getValue() > arr.length / k){
                hasPrint = true;
                ans = ans + entry.getKey() + " ";
            }
        }
        System.out.println( hasPrint == false ? -1 : ans);
    
    }


```

 

#### 面试指南 2 可整合的最大数组长度



```
public static void main(String[] args) throws IOException{
        BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(bf.readLine());
        String[] strs = bf.readLine().split(" ");
        int[] nums = new int[n];
        for(int i = 0; i < n; i++){
            nums[i] = Integer.parseInt(strs[i]);
        }

        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer t0, Integer t1) {
                return t1 - t0;
            }
        });
        queue.add(1);
        HashSet <Integer> set = new HashSet<>();
        for(int i = 0; i < nums.length; i++){
            int max = Integer.MIN_VALUE;
            int min = Integer.MAX_VALUE;
            //set.add(nums[i]);
            for(int j = i; j < nums.length; j++){
                max = Math.max(max,nums[j]);
                min = Math.min(min,nums[j]);
                if(set.contains(nums[j])) break;
                else set.add(nums[j]);
    
                if(max - min == j - i){
                    queue.add(max - min + 1);
                }
    
            }
            set.clear();
        }
            System.out.println(queue.poll());
        //System.out.println(list.get(0));
    }


```



#### 面试指南3 获取二元组 



```
private static String getRes(int[] arr,int k){
        if(arr==null||arr.length<2) return "";

        int left  = 0 , right = arr.length-1;
        StringBuilder sb = new StringBuilder();
        int preleft = 0, preright = right;
    
        while (left<  right){
            if(arr[left] + arr[right] == k){
                sb.append(arr[left]).append(" ").append(arr[right]).append(" ").append("\n");
                preleft = left;
                while (arr[left] == arr[preleft])   left++;
            }else if(arr[left] + arr[right] < k){
                preleft = left;
                while (arr[left] == arr[preleft])   left++;
            }else{
                preright = right;
                while (arr[right] == arr[preright])   right--;
            }
        }
    
        return sb.toString();
    }

```



#### 面试指南 4 获得不重复的三元组



```
public static void printRest(int[] arr,int start, int target, StringBuilder sb){
        int left  = start+1 , right = arr.length-1;
        while (left < right){
            if(arr[left] + arr[right] == target){
                if(arr[left] != arr[left-1])
                    sb.append(arr[start]).append(" ").append(arr[left]).append(" ").append(arr[right]).append(" ").append("\n");
                left++;
                right--;
            }else if(arr[left] + arr[right] < target){
                left++;
            }else{
                right--;
            }
        }

    }
    
    public static void printUniqueTriad(int[] arr, int k){
        StringBuilder sb = new StringBuilder();
        int pre= -1;
        for (int i = 0; i < arr.length-2;i++ ) {
            if(i == 0 || arr[i] != arr[i-1])
                printRest(arr, i , k - arr[i], sb);
    
        }
        System.out.println(sb.toString());
    }


```



#### 面试指南 8 找到累加和中为K 的最大连续数组



```
public static int process(int []nums,int K){
        if(K <=0 ) return  0;
        int left = 0, right = 0;
        int sum = 0, ans = 0;

        while (right< nums.length){
            sum+= nums[right];
            if(sum == K) {
                ans = Math.max(ans,  right - left +1);
                sum -= nums[left++];
            }
            while (left <= right && sum >K){
                sum -=  nums[left++];
            }
            if(sum == K) {
                ans = Math.max(ans,  right - left +1);
                sum -= nums[left++];
            }
    
            right++;
    
        }
        return ans;
    }

```



#### 面试指南 9 累加和为K 的最大数组 数组可正可负  使用前缀和

```
public static int process(int []nums,int K){
        HashMap<Integer ,Integer > map = new HashMap<>();
        int ans = 0;
        int sums =0;
        map.put(0, -1);
        for (int i = 0; i < nums.length; i++) {
            sums+=nums[i];
            if(map.containsKey( sums-K)){
                ans = Math.max(   ans, i - map.get(sums-K));
            }
            if(!map.containsKey(sums))
            {
                map.put(sums, i);
            }
        }
        return ans;
    }


```





#### 面试指南 10 把第九题正数改为1 负数改为-1  ，累加和为0



```
public static int process(int []nums , int K){
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0,-1);
        int sum =0;
        int ans = 0;

        for (int i = 0; i < nums.length; i++) {
            sum+= nums[i];
            if(map.containsKey(sum-K)){
                ans = Math.max(ans, i - map.get(sum-K));
            }
            if(!map.containsKey(sum)) {
                map.put(sum,i);
            }
        }
        return  ans ;
    }
    public static void main(String[] args) {
        Scanner in  = new Scanner(System.in);
        int N = in.nextInt();
        
        int []nums = new int[N];
        for (int i = 0; i < N; i++) {
            int temp = in.nextInt();
            if(temp >0 )  nums[i] =1;
            else if(temp <0) nums[i] = -1;
            else nums[i] = 0;
        }
        System.out.println( process(nums,0));
    }

```



#### 面试指南 11 把0 改为-1， 直接和第10题一样的套路



```
public static void main(String[] args) {
        Scanner in  = new Scanner(System.in);
        int N = in.nextInt();

        int []nums = new int[N];
        for (int i = 0; i < N; i++) {
            int temp = in.nextInt();
    
            if(temp == 0 )  nums[i] =-1;
            else nums[i] = temp;
        }
        System.out.println( process(nums,0));
    }

```

websocket

