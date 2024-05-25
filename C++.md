Here are detailed notes on various C++ concepts along with example code to explain the basic implementation of these topics:

### Overview
- **C++** is a general-purpose programming language that supports procedural, object-oriented, and generic programming.

### Environment Setup
To run C++ code, you need a compiler (e.g., GCC) and a text editor (e.g., Notepad++, Visual Studio Code).

```sh
# Compile and run a C++ program using GCC
g++ hello.cpp -o hello
./hello
```

### Basic Syntax

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```

### Comments in C++

```cpp
// Single-line comment
/*
Multi-line
comment
*/
```

### Data Types

```cpp
int integer = 10;
float floating_point = 10.5;
char character = 'A';
bool boolean = true;
```

### Variable Types

```cpp
int global_variable = 20;

int main() {
    int local_variable = 10;
    cout << global_variable << " " << local_variable << endl;
    return 0;
}
```

### Variable Scope

```cpp
void function() {
    int local_variable = 10;
    cout << "Local: " << local_variable << endl;
}

int main() {
    function();
    return 0;
}
```

### Constants/Literals

```cpp
const int constant_variable = 100;
#define PI 3.14159
```

### Modifier Types

```cpp
unsigned int positive_integer = 100;
signed int negative_integer = -100;
```

### Storage Classes

```cpp
void function() {
    static int static_variable = 0;
    static_variable++;
    cout << static_variable << endl;
}

int main() {
    function();
    function();
    return 0;
}
```

### Operators

```cpp
int a = 10, b = 20;
int sum = a + b;  // Arithmetic operator
bool isEqual = (a == b);  // Relational operator
int andResult = a & b;  // Bitwise operator
```

### Loop Types

```cpp
for (int i = 0; i < 5; i++) {
    cout << i << endl;
}

int j = 0;
while (j < 5) {
    cout << j << endl;
    j++;
}

int k = 0;
do {
    cout << k << endl;
    k++;
} while (k < 5);
```

### Decision-Making Statements

```cpp
int a = 10;
if (a > 0) {
    cout << "Positive" << endl;
} else {
    cout << "Negative" << endl;
}

int b = 2;
switch (b) {
    case 1:
        cout << "One" << endl;
        break;
    case 2:
        cout << "Two" << endl;
        break;
    default:
        cout << "Other" << endl;
        break;
}
```

### Functions

```cpp
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    cout << "Result: " << result << endl;
    return 0;
}
```

### Numbers

```cpp
#include <cmath>
int main() {
    int a = 10, b = 3;
    cout << pow(a, b) << endl;
    return 0;
}
```

### Arrays

```cpp
int arr[5] = {1, 2, 3, 4, 5};
for (int i = 0; i < 5; i++) {
    cout << arr[i] << endl;
}
```

### Strings

```cpp
#include <string>
int main() {
    string str = "Hello";
    cout << str << endl;
    return 0;
}
```

### Pointers

```cpp
int main() {
    int var = 10;
    int *ptr = &var;
    cout << "Value: " << *ptr << endl;
    return 0;
}
```

### References

```cpp
void increment(int &ref) {
    ref++;
}

int main() {
    int a = 10;
    increment(a);
    cout << a << endl;
    return 0;
}
```

### Date and Time

```cpp
#include <ctime>
int main() {
    time_t now = time(0);
    cout << "Current time: " << ctime(&now) << endl;
    return 0;
}
```

### Basic Input/Output

```cpp
int main() {
    int a;
    cout << "Enter a number: ";
    cin >> a;
    cout << "You entered: " << a << endl;
    return 0;
}
```

### Data Structures

```cpp
struct Person {
    string name;
    int age;
};

int main() {
    Person p1 = {"John", 30};
    cout << p1.name << " " << p1.age << endl;
    return 0;
}
```

### Classes and Objects

```cpp
class Box {
public:
    int length;
    void setLength(int l) {
        length = l;
    }
    int getLength() {
        return length;
    }
};

int main() {
    Box box1;
    box1.setLength(10);
    cout << "Box length: " << box1.getLength() << endl;
    return 0;
}
```

### Inheritance

```cpp
class Base {
public:
    void display() {
        cout << "Base class" << endl;
    }
};

class Derived : public Base {
};

int main() {
    Derived obj;
    obj.display();
    return 0;
}
```

### Overloading

```cpp
class Print {
public:
    void show(int i) {
        cout << "Integer: " << i << endl;
    }
    void show(double d) {
        cout << "Double: " << d << endl;
    }
};

int main() {
    Print obj;
    obj.show(5);
    obj.show(5.5);
    return 0;
}
```

### Polymorphism

```cpp
class Base {
public:
    virtual void show() {
        cout << "Base class" << endl;
    }
};

class Derived : public Base {
public:
    void show() override {
        cout << "Derived class" << endl;
    }
};

int main() {
    Base *b;
    Derived d;
    b = &d;
    b->show();
    return 0;
}
```

### Data Abstraction

```cpp
class Abstract {
public:
    virtual void show() = 0;
};

class Derived : public Abstract {
public:
    void show() {
        cout << "Derived class" << endl;
    }
};

int main() {
    Derived d;
    d.show();
    return 0;
}
```

### Data Encapsulation

```cpp
class Encapsulation {
private:
    int x;

public:
    void set(int a) {
        x = a;
    }
    int get() {
        return x;
    }
};

int main() {
    Encapsulation obj;
    obj.set(5);
    cout << "Value: " << obj.get() << endl;
    return 0;
}
```

### Interfaces

```cpp
class Interface {
public:
    virtual void show() = 0;
};

class Implement : public Interface {
public:
    void show() {
        cout << "Implement class" << endl;
    }
};

int main() {
    Implement obj;
    obj.show();
    return 0;
}
```

### Files and Streams

```cpp
#include <fstream>
int main() {
    ofstream outFile("example.txt");
    outFile << "Writing to a file." << endl;
    outFile.close();

    ifstream inFile("example.txt");
    string content;
    while (getline(inFile, content)) {
        cout << content << endl;
    }
    inFile.close();
    return 0;
}
```

### Exception Handling

```cpp
int main() {
    try {
        int a = 10;
        int b = 0;
        if (b == 0) {
            throw "Division by zero!";
        }
        cout << a / b << endl;
    } catch (const char* msg) {
        cerr << "Error: " << msg << endl;
    }
    return 0;
}
```

### Dynamic Memory

```cpp
int main() {
    int *arr = new int[5];
    for (int i = 0; i < 5; i++) {
        arr[i] = i + 1;
    }
    for (int i = 0; i < 5; i++) {
        cout << arr[i] << endl;
    }
    delete[] arr;
    return 0;
}
```

### Namespaces

```cpp
namespace first {
    int var = 10;
}

namespace second {
    int var = 20;
}

int main() {
    cout << first::var << endl;
    cout << second::var << endl;
    return 0;
}
```

### Templates

```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    cout << add<int>(5, 3) << endl;
    cout << add<double>(5.5, 3.3) << endl;
    return 0;
}
```

### Preprocessor

```cpp
#include <iostream>


#define PI 3.14159
using namespace std;

int main() {
    cout << "Value of PI: " << PI << endl;
    return 0;
}
```

### Signal Handling

```cpp
#include <iostream>
#include <csignal>
using namespace std;

void signalHandler(int signum) {
    cout << "Interrupt signal (" << signum << ") received." << endl;
    exit(signum);
}

int main() {
    signal(SIGINT, signalHandler);
    while (1) {
        cout << "Program running..." << endl;
        sleep(1);
    }
    return 0;
}
```

### Multithreading

```cpp
#include <iostream>
#include <pthread.h>
using namespace std;

#define NUM_THREADS 5

void *PrintHello(void *threadid) {
    long tid = (long)threadid;
    cout << "Hello World! Thread ID, " << tid << endl;
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        cout << "main() : creating thread, " << i << endl;
        int rc = pthread_create(&threads[i], NULL, PrintHello, (void *)(long)i);
        if (rc) {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }
    pthread_exit(NULL);
}
```

### STL Tutorial

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> v = {1, 2, 3, 4, 5};
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << endl;
    }
    return 0;
}
```

### Standard Library

```cpp
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main() {
    vector<int> v = {5, 3, 1, 4, 2};
    sort(v.begin(), v.end());
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << endl;
    }
    return 0;
}
```

These code examples provide a comprehensive overview of basic C++ programming concepts, suitable for both beginners and those looking to refresh their knowledge.

here is a detailed explanation of various specialized terms and concepts related to Data Structures and Algorithms (DSA):

### 1. **Memoization**
   - **Definition:** An optimization technique where the results of expensive function calls are stored and reused when the same inputs occur again.
   - **Example:** Storing results of recursive Fibonacci sequence calculations to avoid redundant computations.
     ```cpp
     int fib(int n, vector<int>& memo) {
         if (n <= 1) return n;
         if (memo[n] != -1) return memo[n];
         return memo[n] = fib(n - 1, memo) + fib(n - 2, memo);
     }
     ```

### 2. **Minification**
   - **Definition:** The process of removing all unnecessary characters from source code without changing its functionality. This typically applies to web development for CSS, JavaScript, etc., to reduce file size and improve load times.
   - **Example:** Removing whitespaces, comments, and shortening variable names.

### 3. **Tabulation**
   - **Definition:** A dynamic programming technique where the problem is solved bottom-up and results of subproblems are stored in a table.
   - **Example:** Solving Fibonacci sequence iteratively.
     ```cpp
     int fib(int n) {
         vector<int> dp(n + 1, 0);
         dp[1] = 1;
         for (int i = 2; i <= n; i++) {
             dp[i] = dp[i - 1] + dp[i - 2];
         }
         return dp[n];
     }
     ```

### 4. **Branch and Bound**
   - **Definition:** An algorithm design paradigm for solving combinatorial optimization problems. It involves systematic enumeration of candidate solutions by means of state space search.
   - **Example:** Solving the Travelling Salesman Problem (TSP).

### 5. **Heuristic**
   - **Definition:** Problem-solving methods that use practical, non-optimal solutions to provide good-enough results within a reasonable time frame. Often used for problems where finding an optimal solution is impractical.
   - **Example:** A* algorithm for pathfinding.

### 6. **Lazy Propagation**
   - **Definition:** A technique used in segment trees to delay updates to the segments. This helps in improving the efficiency of range update operations.
   - **Example:** Used in range sum queries and updates.
     ```cpp
     void updateRange(int* tree, int* lazy, int start, int end, int l, int r, int node, int diff) {
         if (lazy[node] != 0) {
             tree[node] += (end - start + 1) * lazy[node];
             if (start != end) {
                 lazy[node * 2 + 1] += lazy[node];
                 lazy[node * 2 + 2] += lazy[node];
             }
             lazy[node] = 0;
         }
         if (start > end || start > r || end < l) return;
         if (start >= l && end <= r) {
             tree[node] += (end - start + 1) * diff;
             if (start != end) {
                 lazy[node * 2 + 1] += diff;
                 lazy[node * 2 + 2] += diff;
             }
             return;
         }
         int mid = (start + end) / 2;
         updateRange(tree, lazy, start, mid, l, r, 2 * node + 1, diff);
         updateRange(tree, lazy, mid + 1, end, l, r, 2 * node + 2, diff);
         tree[node] = tree[node * 2 + 1] + tree[node * 2 + 2];
     }
     ```

### 7. **Dynamic Array**
   - **Definition:** An array that resizes itself automatically when elements are added or removed beyond its capacity. It provides dynamic resizing while maintaining the ability to use array indexing.
   - **Example:** `std::vector` in C++.

### 8. **Sparse Table**
   - **Definition:** A data structure that allows fast queries, especially minimum and maximum queries, in static array problems. It is efficient in terms of preprocessing and query times but uses more space.
   - **Example:** Used for range minimum query (RMQ).
     ```cpp
     void buildSparseTable(int arr[], int n, int sparseTable[][log(n)+1]) {
         for (int i = 0; i < n; i++)
             sparseTable[i][0] = arr[i];
         for (int j = 1; (1 << j) <= n; j++) {
             for (int i = 0; (i + (1 << j) - 1) < n; i++) {
                 sparseTable[i][j] = min(sparseTable[i][j - 1], sparseTable[i + (1 << (j - 1))][j - 1]);
             }
         }
     }
     ```

### 9. **Trie**
   - **Definition:** A tree-like data structure used to store a dynamic set of strings, where each node represents a single character of a string.
   - **Applications:** Efficient for searching words in a dictionary, autocomplete, and spell checking.
   - **Example:**
     ```cpp
     struct TrieNode {
         TrieNode* children[26];
         bool isEndOfWord;
         TrieNode() {
             isEndOfWord = false;
             for (int i = 0; i < 26; i++)
                 children[i] = nullptr;
         }
     };

     class Trie {
     private:
         TrieNode* root;
     public:
         Trie() { root = new TrieNode(); }

         void insert(string key) {
             TrieNode* node = root;
             for (char c : key) {
                 int index = c - 'a';
                 if (!node->children[index])
                     node->children[index] = new TrieNode();
                 node = node->children[index];
             }
             node->isEndOfWord = true;
         }

         bool search(string key) {
             TrieNode* node = root;
             for (char c : key) {
                 int index = c - 'a';
                 if (!node->children[index])
                     return false;
                 node = node->children[index];
             }
             return (node != nullptr && node->isEndOfWord);
         }
     };
     ```

### 10. **Union-Find (Disjoint Set Union - DSU)**
   - **Definition:** A data structure that keeps track of a set of elements partitioned into a number of disjoint (non-overlapping) subsets. It supports two operations: union and find.
   - **Applications:** Network connectivity, Kruskal’s algorithm for MST.
   - **Example:**
     ```cpp
     class UnionFind {
     private:
         vector<int> parent, rank;
     public:
         UnionFind(int n) {
             parent.resize(n);
             rank.resize(n, 0);
             for (int i = 0; i < n; i++)
                 parent[i] = i;
         }

         int find(int u) {
             if (u != parent[u])
                 parent[u] = find(parent[u]);
             return parent[u];
         }

         void unionSets(int u, int v) {
             int rootU = find(u);
             int rootV = find(v);
             if (rootU != rootV) {
                 if (rank[rootU] > rank[rootV])
                     parent[rootV] = rootU;
                 else if (rank[rootU] < rank[rootV])
                     parent[rootU] = rootV;
                 else {
                     parent[rootV] = rootU;
                     rank[rootU]++;
                 }
             }
         }
     };
     ```

### 11. **Segment Tree**
   - **Definition:** A tree data structure used for storing intervals or segments. It allows querying which of the stored segments contain a given point and is useful for answering range queries efficiently.
   - **Applications:** Range sum queries, range minimum queries.
   - **Example:**
     ```cpp
     void buildSegmentTree(int arr[], int segTree[], int low, int high, int pos) {
         if (low == high) {
             segTree[pos] = arr[low];
             return;
         }
         int mid = (low + high) / 2;
         buildSegmentTree(arr, segTree, low, mid, 2 * pos + 1);
         buildSegmentTree(arr, segTree, mid + 1, high, 2 * pos + 2);
         segTree[pos] = segTree[2 * pos + 1] + segTree[2 * pos + 2];
     }

     int rangeQuery(int segTree[], int qlow, int qhigh, int low, int high, int pos) {
         if (qlow <= low && qhigh >= high) return segTree[pos];
         if (qlow > high || qhigh < low) return 0;
         int mid = (low + high) / 2;
         return rangeQuery(segTree, qlow, qhigh, low, mid, 2 * pos + 1) +
                rangeQuery(segTree, qlow, qhigh, mid + 1, high, 2 * pos + 2);
     }
     ```

### 12. **Graph Algorithms**
   - **Breadth-First Search (BFS):** Explores nodes level by level.
   - **Depth-First Search (DFS):** Explores as far as possible along each branch before backtracking.
   - **Dijkstra's Algorithm:** Finds the shortest path in

 a weighted graph with non-negative weights.
   - **Kruskal's Algorithm:** Finds the Minimum Spanning Tree (MST) using the greedy approach.
   - **Prim's Algorithm:** Another approach to find the MST by starting from a single vertex.

### 13. **Sorting Algorithms**
   - **Merge Sort:** A divide and conquer algorithm that splits the array into halves, sorts each half, and merges them.
   - **Quick Sort:** A divide and conquer algorithm that selects a pivot element and partitions the array around the pivot.
   - **Heap Sort:** Converts the array into a binary heap structure and repeatedly extracts the maximum element.
   - **Insertion Sort:** Builds the final sorted array one item at a time by inserting each element in its correct position.

### 14. **Binary Search**
   - **Definition:** A search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search interval in half.
   - **Example:**
     ```cpp
     int binarySearch(vector<int>& arr, int left, int right, int target) {
         while (left <= right) {
             int mid = left + (right - left) / 2;
             if (arr[mid] == target)
                 return mid;
             if (arr[mid] < target)
                 left = mid + 1;
             else
                 right = mid - 1;
         }
         return -1;
     }
     ```

### 15. **Graph Coloring**
   - **Definition:** The assignment of labels (or colors) to the vertices of a graph such that no two adjacent vertices share the same color.
   - **Applications:** Scheduling problems, register allocation.

### 16. **Greedy Algorithms**
   - **Definition:** Algorithms that make the locally optimal choice at each stage with the hope of finding a global optimum.
   - **Examples:** Activity Selection problem, Fractional Knapsack.

### 17. **Topological Sort**
   - **Definition:** A linear ordering of vertices in a directed acyclic graph (DAG) where for every directed edge u -> v, vertex u comes before v in the ordering.
   - **Applications:** Task scheduling.

### 18. **Hashing**
   - **Definition:** A technique used to uniquely identify a specific object from a group of similar objects.
   - **Example:** Hash tables used for efficient data retrieval.
   - **Example:**
     ```cpp
     class HashMap {
     private:
         vector<list<pair<int, int>>> table;
         int size;
         int hashFunction(int key) {
             return key % size;
         }
     public:
         HashMap(int s) : size(s) {
             table.resize(size);
         }

         void insert(int key, int value) {
             int hash = hashFunction(key);
             for (auto& kv : table[hash]) {
                 if (kv.first == key) {
                     kv.second = value;
                     return;
                 }
             }
             table[hash].emplace_back(key, value);
         }

         int get(int key) {
             int hash = hashFunction(key);
             for (auto& kv : table[hash]) {
                 if (kv.first == key)
                     return kv.second;
             }
             return -1;
         }

         void remove(int key) {
             int hash = hashFunction(key);
             table[hash].remove_if([key](const pair<int, int>& kv) { return kv.first == key; });
         }
     };
     ```

These terms and concepts are fundamental in understanding the breadth of Data Structures and Algorithms and their practical applications in computer science.


here is a comprehensive list of basic to advanced algorithms in C++ along with their implementations.

### Basic Algorithms

#### 1. **Linear Search**
```cpp
#include <iostream>
using namespace std;

int linearSearch(int arr[], int n, int key) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == key)
            return i;
    }
    return -1;
}

int main() {
    int arr[] = {2, 4, 0, 1, 9};
    int n = sizeof(arr) / sizeof(arr[0]);
    int key = 1;
    int result = linearSearch(arr, n, key);
    (result == -1) ? cout << "Element not found" : cout << "Element found at index " << result;
    return 0;
}
```

#### 2. **Binary Search**
```cpp
#include <iostream>
using namespace std;

int binarySearch(int arr[], int l, int r, int x) {
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == x) 
            return m;
        if (arr[m] < x)
            l = m + 1;
        else
            r = m - 1;
    }
    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 10;
    int result = binarySearch(arr, 0, n - 1, x);
    (result == -1) ? cout << "Element not present" : cout << "Element found at index " << result;
    return 0;
}
```

### Sorting Algorithms

#### 3. **Bubble Sort**
```cpp
#include <iostream>
using namespace std;

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr) / sizeof(arr[0]);
    bubbleSort(arr, n);
    cout << "Sorted array: \n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
```

#### 4. **Insertion Sort**
```cpp
#include <iostream>
using namespace std;

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6};
    int n = sizeof(arr) / sizeof(arr[0]);
    insertionSort(arr, n);
    cout << "Sorted array: \n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
```

#### 5. **Selection Sort**
```cpp
#include <iostream>
using namespace std;

void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(arr[min_idx], arr[i]);
    }
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    selectionSort(arr, n);
    cout << "Sorted array: \n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
```

#### 6. **Merge Sort**
```cpp
#include <iostream>
using namespace std;

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int arr_size = sizeof(arr) / sizeof(arr[0]);
    mergeSort(arr, 0, arr_size - 1);
    cout << "Sorted array: \n";
    for (int i = 0; i < arr_size; i++)
        cout << arr[i] << " ";
    return 0;
}
```

#### 7. **Quick Sort**
```cpp
#include <iostream>
using namespace std;

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    int arr[] = {10, 7, 8, 9, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    quickSort(arr, 0, n - 1);
    cout << "Sorted array: \n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
```

### Advanced Algorithms

#### 8. **Dijkstra's Algorithm**
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

#define INF 0x3f3f3f3f

typedef pair<int, int> iPair;

void dijkstra(vector<vector<iPair>>& adj, int V, int src) {
    priority_queue<iPair, vector<iPair>, greater<iPair>> pq;
    vector<int> dist(V, INF);
    pq.push(make_pair(0, src));
    dist[src] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (auto x : adj[u]) {
            int v = x.first;
            int weight = x.second;
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.push(make_pair(dist[v], v));
            }
        }
    }

    cout << "Vertex Distance from Source\n";
    for (int i = 0; i < V; ++i)
        cout << i << "\t\t" << dist[i] << "\n";
}

int main() {
    int V = 9;
    vector<vector<iPair>> adj(V);

    adj[0].push_back(make_pair(1, 4));
    adj[0].push_back(make_pair(7, 8));
    adj[1].push_back(make_pair(2, 8));
    adj[1].push_back(make_pair(7, 11));
    adj[2].push_back(make_pair(3, 7));
    adj[2].push_back(make_pair(8, 2));
    adj[2].push_back(make_pair(5, 4));
    adj[3].push_back(make_pair(4, 9));
    adj[3].push_back(make_pair(

5, 14));
    adj[4].push_back(make_pair(5, 10));
    adj[5].push_back(make_pair(6, 2));
    adj[6].push_back(make_pair(7, 1));
    adj[6].push_back(make_pair(8, 6));
    adj[7].push_back(make_pair(8, 7));

    dijkstra(adj, V, 0);

    return 0;
}
```

#### 9. **Prim's Algorithm**
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

typedef pair<int, int> iPair;

void primMST(vector<vector<iPair>>& adj, int V) {
    priority_queue<iPair, vector<iPair>, greater<iPair>> pq;
    int src = 0;
    vector<int> key(V, INT_MAX);
    vector<int> parent(V, -1);
    vector<bool> inMST(V, false);

    pq.push(make_pair(0, src));
    key[src] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;

        for (auto x : adj[u]) {
            int v = x.first;
            int weight = x.second;
            if (!inMST[v] && key[v] > weight) {
                key[v] = weight;
                pq.push(make_pair(key[v], v));
                parent[v] = u;
            }
        }
    }

    for (int i = 1; i < V; ++i)
        cout << parent[i] << " - " << i << "\n";
}

int main() {
    int V = 9;
    vector<vector<iPair>> adj(V);

    adj[0].push_back(make_pair(1, 4));
    adj[0].push_back(make_pair(7, 8));
    adj[1].push_back(make_pair(2, 8));
    adj[1].push_back(make_pair(7, 11));
    adj[2].push_back(make_pair(3, 7));
    adj[2].push_back(make_pair(8, 2));
    adj[2].push_back(make_pair(5, 4));
    adj[3].push_back(make_pair(4, 9));
    adj[3].push_back(make_pair(5, 14));
    adj[4].push_back(make_pair(5, 10));
    adj[5].push_back(make_pair(6, 2));
    adj[6].push_back(make_pair(7, 1));
    adj[6].push_back(make_pair(8, 6));
    adj[7].push_back(make_pair(8, 7));

    primMST(adj, V);

    return 0;
}
```

#### 10. **Kruskal's Algorithm**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
    Graph(int V, int E) {
        this->V = V;
        this->E = E;
    }
};

struct subset {
    int parent;
    int rank;
};

int find(subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

void Union(subset subsets[], int x, int y) {
    int rootX = find(subsets, x);
    int rootY = find(subsets, y);

    if (subsets[rootX].rank < subsets[rootY].rank)
        subsets[rootX].parent = rootY;
    else if (subsets[rootX].rank > subsets[rootY].rank)
        subsets[rootY].parent = rootX;
    else {
        subsets[rootY].parent = rootX;
        subsets[rootX].rank++;
    }
}

bool comp(Edge a, Edge b) {
    return a.weight < b.weight;
}

void KruskalMST(Graph& graph) {
    int V = graph.V;
    vector<Edge> result(V);
    int e = 0;
    int i = 0;

    sort(graph.edges.begin(), graph.edges.end(), comp);

    subset* subsets = new subset[(V * sizeof(subset))];
    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    while (e < V - 1 && i < graph.E) {
        Edge next_edge = graph.edges[i++];
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);

        if (x != y) {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
    }

    for (i = 0; i < e; ++i)
        cout << result[i].src << " -- " << result[i].dest << " == " << result[i].weight << endl;

    delete[] subsets;
}

int main() {
    int V = 4;
    int E = 5;
    Graph graph(V, E);

    graph.edges.push_back({0, 1, 10});
    graph.edges.push_back({0, 2, 6});
    graph.edges.push_back({0, 3, 5});
    graph.edges.push_back({1, 3, 15});
    graph.edges.push_back({2, 3, 4});

    KruskalMST(graph);

    return 0;
}
```

#### 11. **Bellman-Ford Algorithm**
```cpp
#include <iostream>
#include <vector>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
    Graph(int V, int E) {
        this->V = V;
        this->E = E;
    }
};

void BellmanFord(Graph& graph, int src) {
    int V = graph.V;
    int E = graph.E;
    vector<int> dist(V, INT_MAX);
    dist[src] = 0;

    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = graph.edges[j].src;
            int v = graph.edges[j].dest;
            int weight = graph.edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }

    for (int j = 0; j < E; j++) {
        int u = graph.edges[j].src;
        int v = graph.edges[j].dest;
        int weight = graph.edges[j].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
            cout << "Graph contains negative weight cycle" << endl;
    }

    cout << "Vertex Distance from Source" << endl;
    for (int i = 0; i < V; ++i)
        cout << i << "\t\t" << dist[i] << endl;
}

int main() {
    int V = 5;
    int E = 8;
    Graph graph(V, E);

    graph.edges.push_back({0, 1, -1});
    graph.edges.push_back({0, 2, 4});
    graph.edges.push_back({1, 2, 3});
    graph.edges.push_back({1, 3, 2});
    graph.edges.push_back({1, 4, 2});
    graph.edges.push_back({3, 2, 5});
    graph.edges.push_back({3, 1, 1});
    graph.edges.push_back({4, 3, -3});

    BellmanFord(graph, 0);

    return 0;
}
```

#### 12. **Floyd-Warshall Algorithm**
```cpp
#include <iostream>
using namespace std;

#define V 4
#define INF 99999

void printSolution(int dist[][V]);

void floydWarshall(int graph[][V]) {
    int dist[V][V];
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = graph[i][j];

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }

    printSolution(dist);
}

void printSolution(int dist[][V]) {
    cout << "The following matrix shows the shortest distances between every pair of vertices \n";
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF)
                cout << "INF" << "     ";
            else
                cout << dist[i][j] << "     ";
        }
        cout << endl;
    }
}

int main() {
    int graph[V][V] = { {0, 5, INF, 10},
                        {INF, 0, 

3, INF},
                        {INF, INF, 0, 1},
                        {INF, INF, INF, 0} };

    floydWarshall(graph);
    return 0;
}
```

#### 13. **Depth First Search (DFS)**
```cpp
#include <iostream>
#include <list>
using namespace std;

class Graph {
    int V;
    list<int>* adj;
    void DFSUtil(int v, bool visited[]);
public:
    Graph(int V);
    void addEdge(int v, int w);
    void DFS(int v);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::DFSUtil(int v, bool visited[]) {
    visited[v] = true;
    cout << v << " ";

    list<int>::iterator i;
    for (i = adj[v].begin(); i != adj[v].end(); ++i)
        if (!visited[*i])
            DFSUtil(*i, visited);
}

void Graph::DFS(int v) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;
    DFSUtil(v, visited);
}

int main() {
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    cout << "Depth First Traversal (starting from vertex 2) \n";
    g.DFS(2);

    return 0;
}
```

#### 14. **Breadth First Search (BFS)**
```cpp
#include <iostream>
#include <list>
using namespace std;

class Graph {
    int V;
    list<int>* adj;
public:
    Graph(int V);
    void addEdge(int v, int w);
    void BFS(int s);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::BFS(int s) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    list<int> queue;
    visited[s] = true;
    queue.push_back(s);

    list<int>::iterator i;

    while (!queue.empty()) {
        s = queue.front();
        cout << s << " ";
        queue.pop_front();

        for (i = adj[s].begin(); i != adj[s].end(); ++i) {
            if (!visited[*i]) {
                visited[*i] = true;
                queue.push_back(*i);
            }
        }
    }
}

int main() {
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    cout << "Breadth First Traversal (starting from vertex 2) \n";
    g.BFS(2);

    return 0;
}
```

### 15. **Counting Sort**

**Explanation:**
- Counting Sort is a non-comparative sorting algorithm.
- It counts the occurrences of each element in the array.
- It uses the counts to determine the positions of elements in the sorted array.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void countingSort(vector<int>& arr) {
    int maxElement = *max_element(arr.begin(), arr.end());
    vector<int> count(maxElement + 1, 0);

    for (int num : arr)
        count[num]++;

    int index = 0;
    for (int i = 0; i <= maxElement; i++) {
        while (count[i] > 0) {
            arr[index++] = i;
            count[i]--;
        }
    }
}

int main() {
    vector<int> arr = {4, 2, 2, 8, 3, 3, 1};
    countingSort(arr);
    for (int num : arr)
        cout << num << " ";
    return 0;
}
```

### 16. **Radix Sort**

**Explanation:**
- Radix Sort processes each digit of the numbers starting from the least significant to the most significant digit.
- It uses a stable sorting algorithm like Counting Sort for each digit.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int getMax(vector<int>& arr) {
    return *max_element(arr.begin(), arr.end());
}

void countingSort(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];
}

void radixSort(vector<int>& arr) {
    int m = getMax(arr);
    for (int exp = 1; m / exp > 0; exp *= 10)
        countingSort(arr, exp);
}

int main() {
    vector<int> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSort(arr);
    for (int num : arr)
        cout << num << " ";
    return 0;
}
```

### 17. **Bucket Sort**

**Explanation:**
- Bucket Sort distributes elements into several buckets.
- Each bucket is then sorted individually using another sorting algorithm or recursively using Bucket Sort.
- The sorted buckets are concatenated to form the final sorted array.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void bucketSort(vector<float>& arr) {
    int n = arr.size();
    vector<vector<float>> buckets(n);

    for (int i = 0; i < n; i++) {
        int bucketIndex = n * arr[i];
        buckets[bucketIndex].push_back(arr[i]);
    }

    for (int i = 0; i < n; i++)
        sort(buckets[i].begin(), buckets[i].end());

    int index = 0;
    for (int i = 0; i < n; i++) {
        for (float num : buckets[i])
            arr[index++] = num;
    }
}

int main() {
    vector<float> arr = {0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434};
    bucketSort(arr);
    for (float num : arr)
        cout << num << " ";
    return 0;
}
```

### 18. **Topological Sort**

**Explanation:**
- Topological Sort orders vertices in a Directed Acyclic Graph (DAG) such that for any directed edge uv, vertex u comes before v.
- It uses Depth First Search (DFS) to find the order.

**Code:**
```cpp
#include <iostream>
#include <list>
#include <stack>
using namespace std;

class Graph {
    int V;
    list<int>* adj;
    void topologicalSortUtil(int v, bool visited[], stack<int>& Stack);
public:
    Graph(int V);
    void addEdge(int v, int w);
    void topologicalSort();
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::topologicalSortUtil(int v, bool visited[], stack<int>& Stack) {
    visited[v] = true;

    for (int i : adj[v])
        if (!visited[i])
            topologicalSortUtil(i, visited, Stack);

    Stack.push(v);
}

void Graph::topologicalSort() {
    stack<int> Stack;
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    for (int i = 0; i < V; i++)
        if (visited[i] == false)
            topologicalSortUtil(i, visited, Stack);

    while (!Stack.empty()) {
        cout << Stack.top() << " ";
        Stack.pop();
    }
}

int main() {
    Graph g(6);
    g.addEdge(5, 2);
    g.addEdge(5, 0);
    g.addEdge(4, 0);
    g.addEdge(4, 1);
    g.addEdge(2, 3);
    g.addEdge(3, 1);

    cout << "Topological Sort of the given graph \n";
    g.topologicalSort();

    return 0;
}
```

### 19. **KMP (Knuth-Morris-Pratt) Pattern Matching Algorithm**

**Explanation:**
- KMP searches for occurrences of a word within a main text string.
- It uses preprocessing to create a partial match table (LPS array) to avoid unnecessary comparisons.

**Code:**
```cpp
#include <iostream>
#include <vector>
using namespace std;

void computeLPSArray(string pat, int M, vector<int>& lps) {
    int length = 0;
    lps[0] = 0;
    int i = 1;

    while (i < M) {
        if (pat[i] == pat[length]) {
            length++;
            lps[i] = length;
            i++;
        } else {
            if (length != 0) {
                length = lps[length - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(string pat, string txt) {
    int M = pat.length();
    int N = txt.length();
    vector<int> lps(M);

    computeLPSArray(pat, M, lps);

    int i = 0;
    int j = 0;
    while (i < N) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }

        if (j == M) {
            cout << "Found pattern at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < N && pat[j] != txt[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
}

int main() {
    string txt = "ABABDABACDABABCABAB";
    string pat = "ABABCABAB";
    KMPSearch(pat, txt);
    return 0;
}
```

### 20. **Rabin-Karp Algorithm**

**Explanation:**
- Rabin-Karp searches for a pattern in a text using hashing.
- It computes a hash value for the pattern and for each substring of the text.
- It compares the hash values for a quick check of potential matches.

**Code:**
```cpp
#include <iostream>
#include <string>
using namespace std;

#define d 256

void search(string pat, string txt, int q) {
    int M = pat.length();
    int N = txt.length();
    int i, j;
    int p = 0; // hash value for pattern
    int t = 0; // hash value for txt
    int h = 1;

    for (i = 0; i < M - 1; i++)
        h = (h * d) % q;

    for (i = 0; i < M; i++) {
        p = (d * p + pat[i]) % q;
       

        t = (d * t + txt[i]) % q;
    }

    for (i = 0; i <= N - M; i++) {
        if (p == t) {
            for (j = 0; j < M; j++) {
                if (txt[i + j] != pat[j])
                    break;
            }
            if (j == M)
                cout << "Pattern found at index " << i << endl;
        }

        if (i < N - M) {
            t = (d * (t - txt[i] * h) + txt[i + M]) % q;
            if (t < 0)
                t = (t + q);
        }
    }
}

int main() {
    string txt = "GEEKS FOR GEEKS";
    string pat = "GEEK";
    int q = 101;
    search(pat, txt, q);
    return 0;
}
```

### 21. **A* Algorithm**

**Explanation:**
- A* is a pathfinding and graph traversal algorithm.
- It uses heuristics to guide its search.
- It combines features of Dijkstra's Algorithm and Greedy Best-First-Search.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
using namespace std;

struct Node {
    int x, y;
    double cost, heuristic;
    Node* parent;

    Node(int x, int y, double cost, double heuristic, Node* parent = nullptr)
        : x(x), y(y), cost(cost), heuristic(heuristic), parent(parent) {}

    double getScore() const {
        return cost + heuristic;
    }

    bool operator>(const Node& other) const {
        return getScore() > other.getScore();
    }
};

double heuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

vector<pair<int, int>> getNeighbors(int x, int y) {
    return {{x + 1, y}, {x - 1, y}, {x, y + 1}, {x, y - 1}};
}

void aStarSearch(vector<vector<int>>& grid, pair<int, int> start, pair<int, int> goal) {
    priority_queue<Node, vector<Node>, greater<Node>> openSet;
    vector<vector<bool>> closedSet(grid.size(), vector<bool>(grid[0].size(), false));

    Node* startNode = new Node(start.first, start.second, 0, heuristic(start.first, start.second, goal.first, goal.second));
    openSet.push(*startNode);

    while (!openSet.empty()) {
        Node current = openSet.top();
        openSet.pop();

        if (current.x == goal.first && current.y == goal.second) {
            cout << "Path found:\n";
            while (current.parent) {
                cout << "(" << current.x << ", " << current.y << ") <- ";
                current = *current.parent;
            }
            cout << "(" << current.x << ", " << current.y << ")\n";
            return;
        }

        closedSet[current.x][current.y] = true;

        for (auto [nx, ny] : getNeighbors(current.x, current.y)) {
            if (nx >= 0 && ny >= 0 && nx < grid.size() && ny < grid[0].size() && !closedSet[nx][ny] && grid[nx][ny] == 0) {
                double tentativeCost = current.cost + 1;
                Node* neighbor = new Node(nx, ny, tentativeCost, heuristic(nx, ny, goal.first, goal.second), new Node(current));
                openSet.push(*neighbor);
            }
        }
    }

    cout << "No path found.\n";
}

int main() {
    vector<vector<int>> grid = {
        {0, 1, 0, 0, 0},
        {0, 1, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0}
    };

    pair<int, int> start = {0, 0};
    pair<int, int> goal = {4, 4};

    aStarSearch(grid, start, goal);
    return 0;
}
```

### 22. **Ford-Fulkerson Algorithm**

**Explanation:**
- Ford-Fulkerson computes the maximum flow in a flow network.
- It uses augmenting paths to increase the flow.
- It iteratively improves the flow until no more augmenting paths can be found.

**Code:**
```cpp
#include <iostream>
#include <limits.h>
#include <queue>
#include <vector>
using namespace std;

bool bfs(vector<vector<int>>& rGraph, int s, int t, vector<int>& parent) {
    int V = rGraph.size();
    vector<bool> visited(V, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}

int fordFulkerson(vector<vector<int>>& graph, int s, int t) {
    int V = graph.size();
    vector<vector<int>> rGraph = graph;
    vector<int> parent(V);
    int maxFlow = 0;

    while (bfs(rGraph, s, t, parent)) {
        int pathFlow = INT_MAX;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            pathFlow = min(pathFlow, rGraph[u][v]);
        }

        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            rGraph[u][v] -= pathFlow;
            rGraph[v][u] += pathFlow;
        }

        maxFlow += pathFlow;
    }

    return maxFlow;
}

int main() {
    vector<vector<int>> graph = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    cout << "The maximum possible flow is " << fordFulkerson(graph, 0, 5) << endl;
    return 0;
}
```

### 23. **Boyer-Moore String Search Algorithm**

**Explanation:**
- Boyer-Moore is an efficient string searching algorithm.
- It uses preprocessing to create bad character and good suffix tables.
- It shifts the pattern more intelligently than the naive approach.

**Code:**
```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

void badCharHeuristic(string str, int size, vector<int>& badchar) {
    for (int i = 0; i < 256; i++)
        badchar[i] = -1;
    for (int i = 0; i < size; i++)
        badchar[(int)str[i]] = i;
}

void search(string txt, string pat) {
    int m = pat.size();
    int n = txt.size();
    vector<int> badchar(256);

    badCharHeuristic(pat, m, badchar);

    int s = 0;
    while (s <= (n - m)) {
        int j = m - 1;

        while (j >= 0 && pat[j] == txt[s + j])
            j--;

        if (j < 0) {
            cout << "Pattern occurs at index " << s << endl;
            s += (s + m < n) ? m - badchar[txt[s + m]] : 1;
        } else {
            s += max(1, j - badchar[txt[s + j]]);
        }
    }
}

int main() {
    string txt = "ABAAABCD";
    string pat = "ABC";
    search(txt, pat);
    return 0;
}
```

### 24. **Huffman Coding**

**Explanation:**
- Huffman Coding is a compression algorithm for lossless data compression.
- It uses a priority queue to build a Huffman Tree.
- It generates variable-length codes for each character based on their frequencies.

**Code:**
```cpp
#include <iostream>
#include <queue>
#include <vector>
#include <unordered_map>
using namespace std;

struct MinHeapNode {
    char data;
    unsigned freq;
    MinHeapNode* left;
    MinHeapNode* right;

    MinHeapNode(char data, unsigned freq) {
        left = right = nullptr;
        this->data = data;
        this->freq = freq;
    }
};

struct compare {
    bool operator()(

MinHeapNode* l, MinHeapNode* r) {
        return (l->freq > r->freq);
    }
};

void printCodes(struct MinHeapNode* root, string str) {
    if (!root)
        return;

    if (root->data != '$')
        cout << root->data << ": " << str << "\n";

    printCodes(root->left, str + "0");
    printCodes(root->right, str + "1");
}

void HuffmanCodes(char data[], int freq[], int size) {
    struct MinHeapNode *left, *right, *top;
    priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap;

    for (int i = 0; i < size; ++i)
        minHeap.push(new MinHeapNode(data[i], freq[i]));

    while (minHeap.size() != 1) {
        left = minHeap.top();
        minHeap.pop();

        right = minHeap.top();
        minHeap.pop();

        top = new MinHeapNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        minHeap.push(top);
    }

    printCodes(minHeap.top(), "");
}

int main() {
    char arr[] = { 'a', 'b', 'c', 'd', 'e', 'f' };
    int freq[] = { 5, 9, 12, 13, 16, 45 };
    int size = sizeof(arr) / sizeof(arr[0]);

    HuffmanCodes(arr, freq, size);
    return 0;
}
```

### 25. **Prim's Algorithm**

**Explanation:**
- Prim's Algorithm finds the Minimum Spanning Tree (MST) of a weighted undirected graph.
- It starts with a single vertex and grows the MST by adding the smallest edge that connects a vertex in the MST to a vertex outside the MST.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <limits.h>
using namespace std;

int minKey(vector<int>& key, vector<bool>& mstSet, int V) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (!mstSet[v] && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

void printMST(vector<int>& parent, vector<vector<int>>& graph, int V) {
    cout << "Edge \tWeight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << " \t" << graph[i][parent[i]] << " \n";
}

void primMST(vector<vector<int>>& graph, int V) {
    vector<int> parent(V);
    vector<int> key(V, INT_MAX);
    vector<bool> mstSet(V, false);

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet, V);
        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    printMST(parent, graph, V);
}

int main() {
    vector<vector<int>> graph = {
        {0, 2, 0, 6, 0},
        {2, 0, 3, 8, 5},
        {0, 3, 0, 0, 7},
        {6, 8, 0, 0, 9},
        {0, 5, 7, 9, 0}
    };

    primMST(graph, graph.size());
    return 0;
}
```

### 26. **Kruskal's Algorithm**

**Explanation:**
- Kruskal's Algorithm finds the Minimum Spanning Tree (MST) of a weighted undirected graph.
- It sorts all edges in non-decreasing order by weight and adds edges to the MST if they don't form a cycle.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
};

Graph* createGraph(int V, int E) {
    Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    return graph;
}

struct subset {
    int parent;
    int rank;
};

int find(struct subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

void Union(struct subset subsets[], int x, int y) {
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    else {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

int myComp(const void* a, const void* b) {
    Edge* a1 = (Edge*)a;
    Edge* b1 = (Edge*)b;
    return a1->weight > b1->weight;
}

void KruskalMST(Graph* graph) {
    int V = graph->V;
    Edge result[V];
    int e = 0;
    int i = 0;

    qsort(graph->edges.data(), graph->E, sizeof(graph->edges[0]), myComp);

    struct subset* subsets = (struct subset*) malloc(V * sizeof(struct subset));

    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    while (e < V - 1 && i < graph->E) {
        Edge next_edge = graph->edges[i++];
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);

        if (x != y) {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
    }

    cout << "Following are the edges in the constructed MST\n";
    for (i = 0; i < e; ++i)
        cout << result[i].src << " -- " << result[i].dest << " == " << result[i].weight << endl;

    return;
}

int main() {
    int V = 4;
    int E = 5;
    Graph* graph = createGraph(V, E);

    graph->edges = {{0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4}};

    KruskalMST(graph);

    return 0;
}
```

### 27. **Bellman-Ford Algorithm**

**Explanation:**
- Bellman-Ford computes the shortest paths from a single source vertex to all other vertices in a weighted graph.
- It handles negative weight edges.
- It relaxes all edges repeatedly and checks for negative-weight cycles.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <limits.h>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
};

Graph* createGraph(int V, int E) {
    Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    return graph;
}

void printArr(vector<int>& dist, int n) {
    cout << "Vertex Distance from Source\n";
    for (int i = 0; i < n; ++i)
        cout << i << "\t\t" << dist[i] << endl;
}

void BellmanFord(Graph* graph, int src) {
    int V = graph->V;
    int E = graph->E;
    vector<int> dist(V, INT_MAX);

    dist[src] = 0;

    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = graph->edges[j].src;
            int v = graph->edges[j].dest;
            int weight = graph->edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }

    for (int i = 0; i < E; i++) {
        int u = graph->edges[i].src;
        int v = graph->edges[i].dest;
        int weight = graph->edges[i].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
            cout << "Graph contains negative weight cycle" << endl;
    }

    printArr(dist, V);

    return;
}

int main() {
    int V = 5;
    int E = 8;
    Graph* graph = createGraph(V

, E);

    graph->edges = {{0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 2},
                    {3, 2, 5}, {3, 1, 1}, {4, 3, -3}};

    BellmanFord(graph, 0);

    return 0;
}
```

### 28. **Johnson’s Algorithm**

**Explanation:**
- Johnson's Algorithm finds all-pairs shortest paths in a sparse weighted graph.
- It reweights edges using Bellman-Ford and then uses Dijkstra's algorithm on each vertex.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <limits.h>
#include <queue>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
};

Graph* createGraph(int V, int E) {
    Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    return graph;
}

void BellmanFord(Graph* graph, int src, vector<int>& dist) {
    int V = graph->V;
    int E = graph->E;
    dist[src] = 0;

    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = graph->edges[j].src;
            int v = graph->edges[j].dest;
            int weight = graph->edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }

    for (int i = 0; i < E; i++) {
        int u = graph->edges[i].src;
        int v = graph->edges[i].dest;
        int weight = graph->edges[i].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
            cout << "Graph contains negative weight cycle" << endl;
    }
}

void dijkstra(vector<vector<pair<int, int>>>& adj, int src, vector<int>& dist) {
    int V = adj.size();
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    dist[src] = 0;
    pq.push(make_pair(0, src));

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (auto x : adj[u]) {
            int v = x.first;
            int weight = x.second;

            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.push(make_pair(dist[v], v));
            }
        }
    }
}

void Johnson(Graph* graph) {
    int V = graph->V;
    int E = graph->E;

    Graph* newGraph = createGraph(V + 1, E + V);

    for (int i = 0; i < E; i++) {
        newGraph->edges.push_back(graph->edges[i]);
    }

    for (int i = 0; i < V; i++) {
        newGraph->edges.push_back({V, i, 0});
    }

    vector<int> h(V + 1, INT_MAX);
    BellmanFord(newGraph, V, h);

    for (int i = 0; i < E; i++) {
        int u = graph->edges[i].src;
        int v = graph->edges[i].dest;
        graph->edges[i].weight += h[u] - h[v];
    }

    vector<vector<pair<int, int>>> adj(V);

    for (int i = 0; i < E; i++) {
        int u = graph->edges[i].src;
        int v = graph->edges[i].dest;
        int weight = graph->edges[i].weight;
        adj[u].push_back(make_pair(v, weight));
    }

    for (int i = 0; i < V; i++) {
        vector<int> dist(V, INT_MAX);
        dijkstra(adj, i, dist);

        for (int j = 0; j < V; j++) {
            if (dist[j] != INT_MAX) {
                cout << "Distance from " << i << " to " << j << " is " << dist[j] + h[j] - h[i] << endl;
            }
        }
    }
}

int main() {
    int V = 4;
    int E = 5;
    Graph* graph = createGraph(V, E);

    graph->edges = {{0, 1, 1}, {0, 2, 4}, {1, 2, -3}, {1, 3, 2}, {2, 3, 3}};

    Johnson(graph);

    return 0;
}
```

### 29. **Ford-Fulkerson Algorithm**

**Explanation:**
- The Ford-Fulkerson method computes the maximum flow in a flow network.
- It uses depth-first search (DFS) or breadth-first search (BFS) to find augmenting paths and increase the flow until no more augmenting paths are found.

**Code:**
```cpp
#include <iostream>
#include <limits.h>
#include <queue>
#include <vector>
using namespace std;

#define V 6

bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
    bool visited[V] = { false };
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    return false;
}

int fordFulkerson(int graph[V][V], int s, int t) {
    int u, v;
    int rGraph[V][V];

    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
            rGraph[u][v] = graph[u][v];

    int parent[V];
    int max_flow = 0;

    while (bfs(rGraph, s, t, parent)) {
        int path_flow = INT_MAX;

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        max_flow += path_flow;
    }

    return max_flow;
}

int main() {
    int graph[V][V] = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    cout << "The maximum possible flow is " << fordFulkerson(graph, 0, 5) << endl;

    return 0;
}
```

### 30. **Edmonds-Karp Algorithm**

**Explanation:**
- The Edmonds-Karp algorithm is an implementation of the Ford-Fulkerson method using BFS for finding augmenting paths.
- It provides a polynomial time complexity for finding maximum flow.

**Code:**
```cpp
#include <iostream>
#include <limits.h>
#include <queue>
#include <vector>
using namespace std;

#define V 6

bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
    bool visited[V] = { false };
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    return false;
}

int edmondsKarp(int graph[V][V], int s, int t) {
    int u, v;
    int rGraph[V][V];

    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
            rGraph[u][v] = graph[u][v];

    int parent[V];
    int max_flow = 0;

    while (bfs(rGraph, s, t, parent)) {
        int path_flow = INT_MAX;

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        max_flow += path_flow;
    }

    return max_flow;
}

int main() {
    int graph[V][V] = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    cout << "The maximum possible flow is " << edmondsKarp(graph, 0, 5) << endl;

    return 0;
}
```

### 31. **Hopcroft-Karp Algorithm**

**Explanation:**
- The Hopcroft-Karp algorithm finds the maximum cardinality matching in a bipartite graph.
- It uses BFS and DFS to find augmenting paths.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

#define NIL 0
#define INF INT_MAX

class BipartiteGraph {
    int m, n;
    vector<int>* adj;
    int* pairU, * pairV, * dist;

public:
    BipartiteGraph(int m, int n);
    void addEdge(int u, int v);
    bool bfs();
    bool dfs(int u);
    int hopcroftKarp();
};

BipartiteGraph::BipartiteGraph(int m, int n) {
    this->m = m;
    this->n = n;
    adj = new vector<int>[m + 1];
    pairU = new int[m + 1];
    pairV = new int[n + 1];
    dist = new int[m + 1];
}

void BipartiteGraph::addEdge(int u, int v) {
    adj[u].push_back(v);
}

bool BipartiteGraph::bfs() {
    queue<int> Q;

    for (int u = 1; u <= m; u++) {
        if (pairU[u] == NIL) {
            dist[u] = 0;
            Q.push(u);
        } else {
            dist[u] = INF;
        }
    }

    dist[NIL] = INF;

    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();

        if (dist[u] < dist[NIL]) {
            for (int i : adj[u]) {
                int v = i;
                if (dist[pairV[v]] == INF) {
                    dist[pairV[v]] = dist[u] + 1;
                    Q.push(pairV[v]);
                }
            }
        }
    }

    return (dist[NIL] != INF);
}

bool BipartiteGraph::dfs(int u) {
    if (u != NIL) {
        for (int i : adj[u]) {
            int v = i;
            if (dist[pairV[v]] == dist[u] + 1) {
                if (dfs(pairV[v])) {
                    pairV[v] = u;
                    pairU[u] = v;
                    return true;
                }
            }
        }

        dist[u] = INF;
        return false;
    }
    return true;
}

int BipartiteGraph::hopcroftKarp() {
    for (int u = 0; u <= m; u++) pairU[u] = NIL;
    for (int v = 0; v <= n; v++) pairV[v] = NIL;

    int result = 0;

    while (bfs()) {
        for (int u = 1; u <= m; u++)
            if (pairU[u] == NIL && dfs(u))
                result++;
    }
    return result;
}

int main() {
    int m = 4, n = 4;
    BipartiteGraph g(m, n);
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 1);
    g.addEdge(3, 2);
    g.addEdge(4, 2);
    g.addEdge(4, 4);

    cout << "Size of maximum matching is " << g.hopcroftKarp() << endl;

    return 0;
}
```

### 32. **Boruvka’s Algorithm**

**Explanation:**
- Boruvka's algorithm is a parallel algorithm for finding the Minimum Spanning Tree (MST

) of a graph.
- It repeatedly adds the shortest edge from each component to the MST until all vertices are connected.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
};

Graph* createGraph(int V, int E) {
    Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    return graph;
}

struct subset {
    int parent;
    int rank;
};

int find(subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

void Union(subset subsets[], int x, int y) {
    int rootX = find(subsets, x);
    int rootY = find(subsets, y);

    if (subsets[rootX].rank < subsets[rootY].rank)
        subsets[rootX].parent = rootY;
    else if (subsets[rootX].rank > subsets[rootY].rank)
        subsets[rootY].parent = rootX;
    else {
        subsets[rootY].parent = rootX;
        subsets[rootX].rank++;
    }
}

void boruvkaMST(Graph* graph) {
    int V = graph->V, E = graph->E;
    Edge* edge = graph->edges.data();

    subset* subsets = new subset[V];
    int* cheapest = new int[V];

    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
        cheapest[v] = -1;
    }

    int numTrees = V;
    int MSTweight = 0;

    while (numTrees > 1) {
        for (int v = 0; v < V; ++v) {
            cheapest[v] = -1;
        }

        for (int i = 0; i < E; i++) {
            int set1 = find(subsets, edge[i].src);
            int set2 = find(subsets, edge[i].dest);

            if (set1 == set2)
                continue;

            else {
                if (cheapest[set1] == -1 || edge[cheapest[set1]].weight > edge[i].weight)
                    cheapest[set1] = i;

                if (cheapest[set2] == -1 || edge[cheapest[set2]].weight > edge[i].weight)
                    cheapest[set2] = i;
            }
        }

        for (int i = 0; i < V; i++) {
            if (cheapest[i] != -1) {
                int set1 = find(subsets, edge[cheapest[i]].src);
                int set2 = find(subsets, edge[cheapest[i]].dest);

                if (set1 == set2)
                    continue;
                MSTweight += edge[cheapest[i]].weight;
                cout << "Edge " << edge[cheapest[i]].src << "-" << edge[cheapest[i]].dest << " included in MST\n";
                Union(subsets, set1, set2);
                numTrees--;
            }
        }
    }

    cout << "Weight of MST is " << MSTweight << endl;
    return;
}

int main() {
    int V = 4;
    int E = 5;
    Graph* graph = createGraph(V, E);

    graph->edges = { {0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4} };

    boruvkaMST(graph);

    return 0;
}
```

### 33. **Floyd-Warshall Algorithm**

**Explanation:**
- The Floyd-Warshall algorithm finds the shortest paths between all pairs of vertices in a weighted graph.
- It uses dynamic programming to update the shortest paths iteratively.

**Code:**
```cpp
#include <iostream>
#include <vector>
using namespace std;

#define INF 99999
#define V 4

void printSolution(int dist[][V]) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF)
                cout << "INF ";
            else
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }
}

void floydWarshall(int graph[][V]) {
    int dist[V][V];

    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = graph[i][j];

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }

    printSolution(dist);
}

int main() {
    int graph[V][V] = { {0, 5, INF, 10},
                        {INF, 0, 3, INF},
                        {INF, INF, 0, 1},
                        {INF, INF, INF, 0} };

    floydWarshall(graph);
    return 0;
}
```

### 34. **Prim’s Algorithm**

**Explanation:**
- Prim's algorithm finds the Minimum Spanning Tree (MST) for a connected weighted graph.
- It starts from an arbitrary vertex and grows the MST by adding the smallest edge from the tree to a vertex not in the tree.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

#define V 5

int minKey(int key[], bool mstSet[]) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (mstSet[v] == false && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

void printMST(int parent[], int graph[V][V]) {
    cout << "Edge \tWeight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << " \t" << graph[i][parent[i]] << " \n";
}

void primMST(int graph[V][V]) {
    int parent[V];
    int key[V];
    bool mstSet[V];

    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet);
        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    printMST(parent, graph);
}

int main() {
    int graph[V][V] = { {0, 2, 0, 6, 0},
                        {2, 0, 3, 8, 5},
                        {0, 3, 0, 0, 7},
                        {6, 8, 0, 0, 9},
                        {0, 5, 7, 9, 0} };

    primMST(graph);
    return 0;
}
```

### 35. **Kruskal’s Algorithm**

**Explanation:**
- Kruskal's algorithm finds the Minimum Spanning Tree (MST) for a connected weighted graph.
- It sorts all the edges by weight and adds them one by one, skipping those that form a cycle.

**Code:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
};

Graph* createGraph(int V, int E) {
    Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    return graph;
}

struct subset {
    int
