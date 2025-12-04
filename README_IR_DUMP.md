# 如何在没有 CUDA Driver 的情况下编译并 Dump IR

## 问题背景

如果你的 CUDA driver 版本不足，无法调用 C 库，但你只想获取 IR dump 而不实际运行 kernel，可以使用本方案。

## 解决方案

本项目提供了一个 mock 的 `_cext.py` 模块，它提供了必要的类和函数定义，但不实际调用 CUDA C 库。这样就可以完成编译和 IR 生成，而无需真实的 CUDA driver。

## 修改内容

### 1. 创建 Mock `_cext.py` 模块

在 `cuda/tile/_cext.py` 中创建了以下 mock 实现：

- `TileDispatcher`: kernel 装饰器的基类
- `TileContext`: 编译上下文
- `get_compute_capability()`: 返回默认的 SM 架构（SM 8.0）
- `ArraySpecialization`: 数组类型分析类
- `launch()`: 抛出错误，因为无法实际启动 kernel

### 2. 修改 `example.py`

示例代码展示了如何：

1. 定义 kernel 函数（使用 `@ct.kernel` 装饰器）
2. 创建 `MockTensor` 对象（实现 `__cuda_array_interface__` 协议）
3. 调用编译函数生成 IR
4. 将 IR dump 到文件

## 使用方法

### 运行示例

```bash
python example.py
```

### 输出文件

编译成功后，IR 文件将保存在 `./ir_artifacts` 目录下：

- `softmax.cutileir`: CuTile IR（高级 IR，人类可读）
- `softmax.cutile`: Bytecode（序列化的 IR，二进制格式）
- `softmax.cuda_tile.mlir`: MLIR（如果 `cuda.tile_internal` 可用）

### CuTile IR 示例

```
func @softmax(x: Array[float32,(?,?):(128,1)], out: Array[float32,(?,?):(128,1)], r.0: int32):
    $token: Token = make_token()
    $5: const int32 = typed_const(value=0)
    $11: int32 = tile_bid(axis=0)
    $13: Tuple[int32,int32] = build_tuple(items=($5, $11))
    $18: Tile[float32,(512,8)], $token.0: Token = tile_load_token_ordered(...)
    $114: Tile[float32,(8)] = tile_reduce(x=$18, fn="max", axis=0, ...)
    ...
```

## 自定义配置

### 修改目标架构

在 `cuda/tile/_cext.py` 中修改 `get_compute_capability()` 函数：

```python
def get_compute_capability():
    # 修改为你的目标架构
    return (8, 0)  # SM 8.0 (Ampere)
    # return (7, 5)  # SM 7.5 (Turing)
    # return (8, 6)  # SM 8.6 (Ampere)
    # return (8, 9)  # SM 8.9 (Ada Lovelace)
    # return (9, 0)  # SM 9.0 (Hopper)
```

### 修改 IR dump 位置

在 `example.py` 中修改环境变量：

```python
os.environ['CUDA_TILE_DUMP_TILEIR'] = './my_ir_artifacts'
os.environ['CUDA_TILE_DUMP_BYTECODE'] = './my_ir_artifacts'
```

## 编写自己的 Kernel

### 基本模板

```python
import cuda.tile as ct

@ct.kernel
def my_kernel(input_array, output_array, constant_param: ct.Constant):
    # 使用常量参数
    tile = ct.load(input_array, (ct.bid(0), 0), (256, constant_param))
    
    # 进行计算
    result = ct.exp(tile)
    
    # 存储结果
    ct.store(output_array, (ct.bid(0), 0), result)
```

### 重要注意事项

1. **常量参数**: 如果参数需要在编译时确定（如 tile 形状），必须使用 `ct.Constant` 注解
2. **MockTensor**: 必须实现 `__cuda_array_interface__` 协议
3. **形状**: `ct.load` 和 `ct.store` 的形状参数必须是常量

### 完整示例

```python
import cuda.tile as ct
from cuda.tile._compile import _get_final_ir
from cuda.tile._cext import default_tile_context

@ct.kernel
def my_kernel(x, out, tile_size: ct.Constant):
    tile = ct.load(x, (ct.bid(0), 0), (tile_size, tile_size))
    result = ct.exp(tile)
    ct.store(out, (ct.bid(0), 0), result)

# 创建 mock tensor
class MockTensor:
    def __init__(self, shape):
        self.shape = shape
        self.__cuda_array_interface__ = {
            'shape': shape,
            'typestr': '<f4',  # float32
            'data': (0, False),
            'version': 3,
        }

x = MockTensor((1024, 1024))
out = MockTensor((1024, 1024))

# 编译并生成 IR
func_ir = _get_final_ir(my_kernel._pyfunc, (x, out, 256), default_tile_context)
print(func_ir.to_string(include_loc=False))
```

## 限制

1. **无法实际运行**: 由于没有真实的 CUDA driver，无法调用 `ct.launch()` 来实际运行 kernel
2. **无法生成 cubin**: `compile_cubin()` 需要 `tileiras` 编译器，如果没有安装会失败
3. **MLIR 转换**: 需要 `cuda.tile_internal` 模块才能将 bytecode 转换为 MLIR

## 故障排除

### 错误: "Expected a constant integer tuple"

确保 `ct.load` 和 `ct.store` 的形状参数是常量。如果使用参数，需要添加 `ct.Constant` 注解。

### 错误: "Python value ... is not supported"

确保你的 tensor 对象实现了 `__cuda_array_interface__` 协议。

### 错误: "cannot import name 'XXX' from 'cuda.tile._cext'"

检查 `cuda/tile/_cext.py` 是否包含了所有必要的类和函数定义。

## 总结

通过创建 mock `_cext.py` 模块，你可以：

✅ 编译 cuda.tile kernel  
✅ 生成和查看 CuTile IR  
✅ 生成 bytecode  
✅ 在没有 CUDA driver 的环境中工作  

❌ 无法实际运行 kernel  
❌ 无法生成 cubin（除非安装了 tileiras 编译器）
