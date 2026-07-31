# CommMemType

## 功能说明

内存物理位置类型。

## 定义原型

```c
typedef enum {
    COMM_MEM_TYPE_INVALID = -1,   /* 无效的内存类别 */
    COMM_MEM_TYPE_DEVICE = 0,     /* Device侧内存（如NPU等） */
    COMM_MEM_TYPE_HOST = 1,       /* Host侧内存 */
} CommMemType;
```
